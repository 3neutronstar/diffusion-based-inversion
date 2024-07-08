import os
import glob
import sys
import logging
from PIL import Image
import torch
import resnet
import argparse
from utils import AverageMeter, accuracy, fix_seed, gather_distributed
from logger import convert_secs2time, set_logging_defaults, create_distributed_logging
import tqdm
from diffusers import DDIMScheduler

def get_args(args):
    
    parser = argparse.ArgumentParser(description='DDPM Inversion')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset load')
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument('--batch_size', type=int, default=60, help='batch size (default: 256)')
    parser.add_argument('--num_steps', type=int, default=1000, help='number of steps (default: 1000)')
    parser.add_argument('--prefix', type=str, default='', help='prefix of file')
    parser.add_argument('--guidance_weight', type=float, default=40., help='CLSF guidance init weight')
    parser.add_argument('--linear_guidance_end_weight', type=float, default=10., help='linear scheduler guidance end weight')
    parser.add_argument('--schedule_type', type=str, default='exp', choices=['exp','linear'], help='schedule type of guidance weight')
    parser.add_argument('--step_type', type=str, default='scale', choices=['reduce','scale'], help='reduce or scale for approximation')
    parser.add_argument('--decay_rate', type=float, default=0.999, help='decay rate of scheduler')
    parser.add_argument('--guidance_num_steps', type=int, default=0, help='num guidance step')
    parser.add_argument('--clipping', type=float, default=5e-2, help='clipping factor of guidance')
    parser.add_argument('--gen_type', type=str, default='uniform', choices=['uniform','batch'], help='generation type')
    parser.add_argument('--gen_cond', type=float, default=0.9, help='generation condition of CE')
    parser.add_argument('--gpu_ids', default='0',
                        type=str, help=' ex) 0,1,2')
    parser.add_argument('--ddp', action='store_true', help='use ddp')
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank')
    
    args = parser.parse_args()

    return args


def main(args):
    from our_model_step import OurDDPMScheduler
    from diffusers import UNet2DModel
    import torch.utils
    import torchvision.utils as vutils
    import torch.distributed as dist
    from torch.utils.tensorboard import SummaryWriter
    from torch.nn.parallel import DistributedDataParallel

    args.prefix = f'./results/{args.prefix}'


    fix_seed(args.seed)
    if args.local_rank == 0:
        print("seed fixed to ",args.seed)

    if args.dataset == 'cifar10':
        num_classes=10
        scheduler = OurDDPMScheduler.from_pretrained("google/ddpm-cifar10-32")
        model = UNet2DModel.from_pretrained("google/ddpm-cifar10-32").to("cuda")
        classifier = resnet.ResNet34(num_classes=10)
        classifier.load_state_dict(torch.load('./resnet34_cifar10_9557.pt'))

        new_scheduler = DDIMScheduler.from_pretrained("google/ddpm-cifar10-32")
        new_scheduler.set_timesteps(5)
    elif args.dataset == 'cifar100':
        import timm
        classifier = timm.create_model("resnet34_cifar100", pretrained=True)
        scheduler = OurDDPMScheduler.from_pretrained("google/ddpm-cifar100-32")
        model = UNet2DModel.from_pretrained("google/ddpm-cifar100-32").to("cuda")
        new_scheduler = DDIMScheduler.from_pretrained("google/ddpm-cifar100-32")
        new_scheduler.set_timesteps(5)


    model.eval()
    scheduler.set_timesteps(args.num_steps)

    # track confidence score of each class
    classifier.eval()
    if args.ddp:
        if int(torch.__version__.split('.')[0])>=2: 
            # pytorch 2.x with torchrun
            model.cuda(args.local_rank)
            model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device = args.local_rank)
            classifier.cuda(args.local_rank)
            classifier = DistributedDataParallel(classifier, device_ids=[args.local_rank], output_device = args.local_rank)
        else: 
            model.cuda(args.local_rank)
            model = DistributedDataParallel(model, device_ids=[args.local_rank])
            classifier.cuda(args.local_rank)
            classifier = DistributedDataParallel(classifier, device_ids=[args.local_rank])

    # define prefix
    args.prefix = args.prefix + f'{args.step_type}step_'
    if args.schedule_type =='exp':
        args.prefix = args.prefix + f'gs{args.guidance_num_steps}_gw{args.guidance_weight:.1f}_schedule_{args.schedule_type}_decay_{args.decay_rate}'
    elif args.schedule_type =='linear':
        args.prefix = args.prefix + f'gs{args.guidance_num_steps}_gw{args.guidance_weight:.1f}_schedule_{args.schedule_type}_endgw{args.linear_guidance_end_weight}'

    args.prefix = args.prefix + f'_clip{args.clipping}'
    args.prefix = args.prefix + f'_cond{args.gen_cond}'

    # mkdir prefix
    
    if args.local_rank ==0:
        if not os.path.exists(args.prefix):
            os.makedirs(args.prefix, exist_ok=True)
        # logger = get_logger(args)
        writer = SummaryWriter(f'{args.prefix}')

    bs = args.batch_size
    if args.ddp:
        sample_size = model.module.config.sample_size
    else:
        sample_size = model.config.sample_size

    if args.gen_type == 'uniform':
        labels = torch.LongTensor( list(range(0,10))* (bs//10) + list(range(0,bs%10)) ).to('cuda')
        noise = torch.randn((bs, 3, sample_size, sample_size)).to("cuda")

    no_reduce_criterion=torch.nn.CrossEntropyLoss(reduction='none')
            
    def exponential_decay_list(init_weight, decay_rate, num_steps):
        weights = [init_weight * (decay_rate ** i) for i in range(num_steps)]
        return torch.tensor(weights)

    if args.schedule_type =='exp':
        sync_scheduler=exponential_decay_list(
                    init_weight=args.guidance_weight,
                    decay_rate=args.decay_rate,
                    num_steps=args.num_steps
                )
    elif args.schedule_type =='linear':
        sync_scheduler = torch.linspace(args.guidance_weight, 10, args.num_steps)
    else:
        raise NotImplementedError



    for param in classifier.parameters():
        param.requires_grad = False
    for param in model.parameters():
        param.requires_grad = False

    @torch.enable_grad()
    def cond_fn(x, t, y=None, classifier=None, orig_inputs=None):
        # if t >=args.guidance_num_steps:
        #     return torch.zeros_like(x)
        nonlocal new_scheduler
        loc = 0
        x_in = orig_inputs.detach().requires_grad_(True)
        x_list = [x_in]
        if args.step_type == 'scale':
            unit_step = int(t) // 5
            new_scheduler.set_timesteps(1000//unit_step)
        else:
            new_scheduler.set_timesteps(5)
            
        ts=[]
        steps=0
        for new_sample_t in new_scheduler.timesteps:
            if new_sample_t >= t or steps >= 5:
                continue
            else:
                steps+=1
            #     new_sample_t = int(round(float(t)/200)*200)
            noisy_residual = model(x_list[-1], new_sample_t).sample
            new_sample = new_scheduler.step(noisy_residual, new_sample_t, x_list[-1]).prev_sample
            x_list.append(new_sample)
            # ts.append(new_sample_t)
        assert y is not None
        # with torch.enable_grad():
        logits  = classifier(new_sample) #/ 2.0 # 0.1
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        selected = log_probs[range(len(logits)), y.view(-1)]

        # scheduling_weight = sync_scheduler[t.item()]
        scheduling_weight = sync_scheduler[1000-t.item()-1]
        with torch.no_grad():
            loss = no_reduce_criterion(logits, y).mean()
        # save images
        if t % 25 == 0 and args.gen_type == 'uniform' and args.local_rank==0:
            vutils.save_image(x_list[-1].detach(),os.path.join(f'{args.prefix}',f'intermediate_image_approx_{t}.png'),normalize=True,scale_each=True,nrow=10)
            print(" scheduler weight : {:.4e} | Loss {:.4e}".format(scheduling_weight,loss.item()))
            # print(ts)
        
        grads = torch.autograd.grad(selected.sum(), x_list[0])[0]# * 10.0
        # grads = torch.autograd.grad(loss, x_list[0])[0]# * 10.0


        # value clamping
        if args.clipping >0:
            clipped_grads = torch.clamp(grads, -args.clipping, args.clipping)
        else:
            pass
            
        # max_norm= args.clipping
        # # max_norm =1.0/2
        # batch_norm = torch.norm(grads.view(grads.shape[0],-1).detach(),dim=1)
        # indices=batch_norm>max_norm
        # grads[indices] = (grads * (max_norm / (batch_norm + 1e-6)).view(-1,1,1,1))[indices]

        # print((pre_grads==grads).sum())
        # exit()
        if args.gen_type == 'uniform' and args.local_rank==0:
            writer.add_scalar('clsf_guiding_loss',loss.item(),1000-t.item())
            writer.add_scalar('scheduler_weight',scheduling_weight,1000-t.item())
            writer.add_scalar('clipped_clsf_guiding_l2_norm',torch.norm(clipped_grads.view(grads.shape[0],-1).detach(),dim=1).mean().item(), 1000-t.item())
            writer.add_scalar('orig_clsf_guiding_l2_norm',torch.norm(grads.view(grads.shape[0],-1).detach(),dim=1).mean().item(), 1000-t.item())
            writer.add_scalar('weighted_clsf_guiding_l2_norm',torch.norm(clipped_grads.view(grads.shape[0],-1).detach(),dim=1).mean().item()*scheduling_weight, 1000-t.item())

        return clipped_grads * scheduling_weight #, None


    if args.gen_type == 'uniform':
        inputs = noise
        for t in scheduler.timesteps:
            if t.item() >=args.guidance_num_steps:
                # with torch.no_grad():
                noisy_residual = model(inputs, t).sample
                prev_noisy_sample = scheduler.step(noisy_residual, t, inputs, cond_fn=cond_fn,
                classifier=classifier,
                labels=labels).prev_sample
                inputs=prev_noisy_sample
                B = inputs.shape[0]
                with torch.no_grad():
                    outputs = classifier(inputs)
                    loss = no_reduce_criterion(outputs, labels).mean()
            else:
                with torch.no_grad():
                    noisy_residual = model(inputs, t).sample
                    prev_noisy_sample = scheduler.step(noisy_residual, t, inputs).prev_sample
                inputs = prev_noisy_sample
                with torch.no_grad():
                    outputs = classifier(inputs)
                    loss = no_reduce_criterion(outputs, labels).mean()
                    
            if t.item() % 25 == 0 and args.local_rank == 0:
                print("{}step | Loss {:.3e}".format(t.item(),loss.item()))
                vutils.save_image(inputs.detach(),os.path.join(f'{args.prefix}',f'intermediate_image_orig_{t}.png'),normalize=True,scale_each=True,nrow=10)
            
            # track confidence score of each class
            if args.local_rank == 0:
                with torch.no_grad():
                    logits= torch.softmax(outputs,dim=1)
                    for c in range(10):
                        class_indices=torch.where(labels==c)[0]
                        if len(class_indices) == 0:
                            continue
                        class_outputs=logits[class_indices][:,c]
                        writer.add_scalar(f'confidence/class_{c}',class_outputs.mean().item(),1000-t.item())
                        for idx in range(class_outputs.shape[0]):
                            writer.add_scalar(f'class_confidence/class_{c}_each_{idx}',class_outputs[idx].item(),1000-t.item())
                    # loss track
                    writer.add_scalar('loss',loss.mean().item(),1000-t.item())
                
            if t.item() % 10 == 0 and args.local_rank == 0:
                print(t.item(),"-th step")

        with torch.no_grad():
            outputs = classifier(inputs)
            loss = no_reduce_criterion(outputs, labels)
            loss_target = loss.mean().item()
            if args.local_rank == 0:
                print("target loss : ",loss_target)
        if args.local_rank == 0:
            vutils.save_image(inputs.detach(),os.path.join(f'{args.prefix}','final.png'),normalize=True,scale_each=True,nrow=10)

            # save inputs to the each class directory (if we don't have then generate folder)
            for i in range(10):
                if not os.path.exists(os.path.join(f'{args.prefix}','class_{}'.format(i))):
                    os.makedirs(os.path.join(f'{args.prefix}','class_{}'.format(i)))
            cls_idx = torch.zeros(10)
            for idx, img in enumerate(inputs):
                # save img
                img = (img / 2 + 0.5).clamp(0, 1)
                img = img.detach().cpu().permute(1, 2, 0).numpy()
                img = Image.fromarray((img * 255).round().astype("uint8"))
                cls_idx[labels[idx].item()] += 1
                img.save(os.path.join(f'{args.prefix}',f'class_{labels[idx].item()}','{}_{}.png'.format(cls_idx[labels[idx].item()].item(),loss[idx].item())))

    elif args.gen_type =='batch':
        accs=[]

        if args.local_rank == 0:
            for i in range(10):
                if not os.path.exists(os.path.join(f'{args.prefix}','class_{}'.format(i))):
                    os.makedirs(os.path.join(f'{args.prefix}','class_{}'.format(i)))
                accs.append(AverageMeter(f'class_{i}Acc',':.2f'))
        import glob
        for c in range(10): # class
            files = glob.glob(os.path.join(f'{args.prefix}','class_{}'.format(c),'*.png'))
            num_files = len(files)
            if args.local_rank == 0:
                print("class {} | num_files : {}".format(c,num_files))
            # samples_per_class=100
            samples_per_class=5000

            class_iter=0
            if args.ddp:
                dist.barrier()

            while num_files <= samples_per_class:
                labels = torch.LongTensor([c]*bs).to('cuda')
                noise = torch.randn((bs, 3, sample_size, sample_size)).to("cuda")
                inputs = noise
                for t in tqdm.tqdm(scheduler.timesteps):
                    if t.item() >=args.guidance_num_steps:
                        # with torch.no_grad():
                        noisy_residual = model(inputs, t).sample
                        prev_noisy_sample = scheduler.step(noisy_residual, t, inputs, cond_fn=cond_fn,
                        classifier=classifier,
                        labels=labels).prev_sample
                        inputs=prev_noisy_sample
                        B = inputs.shape[0]
                        with torch.no_grad():
                            outputs = classifier(inputs)
                            loss = no_reduce_criterion(outputs, labels)
                        
                    else:
                        with torch.no_grad():
                            noisy_residual = model(inputs, t).sample
                            prev_noisy_sample = scheduler.step(noisy_residual, t, inputs).prev_sample
                        inputs = prev_noisy_sample
                        with torch.no_grad():
                            outputs = classifier(inputs)
                            loss = no_reduce_criterion(outputs, labels)
                    # if t.item() % 25 == 0:
                    #     print("{} step | Loss {:.3e}".format(t.item(),loss.mean().item()))
                # save the image when confidence is over 0.9
                if args.ddp:
                    dist.barrier()
                    with torch.no_grad():
                        logits = torch.softmax(outputs,dim=1)
                        logits = gather_distributed(logits)[0]
                        inputs = gather_distributed(inputs)[0]
                        loss = gather_distributed(loss)[0]
                        labels = gather_distributed(labels)[0]

                if args.local_rank == 0:
                    # get file list from the directory and count the number of files
                    files = glob.glob(os.path.join(f'{args.prefix}','class_{}'.format(c),'*.png'))
                    num_files = len(files)
                    with torch.no_grad():
                        count = num_files
                        for idx in range(logits.shape[0]):
                            if logits[idx][c] > args.gen_cond and count <= samples_per_class:
                                img = (inputs[idx] / 2 + 0.5).clamp(0, 1)
                                img = img.detach().cpu().permute(1, 2, 0).numpy()
                                img = Image.fromarray((img * 255).round().astype("uint8"))
                                img.save(os.path.join(f'{args.prefix}',f'class_{c}','{}_{:.4e}.png'.format(count,loss[idx].item())))
                                count +=1
                        acc = accuracy(logits, labels, topk=(1,))
                        accs[c].update(acc[0].item(),bs)
                        num_files = count
                        print("class {} | num_files : {} | acc: {}".format(c,num_files,accs[c].avg))
                        # print("class {} | acc: {}".format(c,accs[c].avg))
                    writer.add_scalar('class_{}/acc_cum'.format(c),accs[c].avg,class_iter)
                    writer.add_scalar('class_{}/acc'.format(c),acc[0].item(),class_iter)
                    writer.add_scalar('class_{}/num_files'.format(c),num_files,class_iter)
                    writer.add_scalar('class_{}/loss'.format(c),loss.mean().item(),class_iter)
                class_iter +=1
                # hold gpu and stay for waiting ddp
                if args.ddp:
                    dist.barrier()
                    files = glob.glob(os.path.join(f'{args.prefix}','class_{}'.format(c),'*.png'))
                    num_files = len(files)


def set_ddp(args):
    import torch.distributed as dist
    ##################
    use_cuda = torch.cuda.is_available()
    device = torch.device(
        "cuda" if use_cuda else "cpu")
    if device.type == "cuda":
        if not args.ddp:
            # when not using ddp
            args.local_rank = 0
        elif int(torch.__version__.split('.')[0])>=2:
            # pytorch 2.0 ddp with torchrun
            if int(os.environ["LOCAL_RANK"])==0:
                print("pytorch 2.x with torchrun, local_rank argument is ignored")
            print("local rank is set to: " + (os.environ["LOCAL_RANK"]))
            args.local_rank = int(os.environ["LOCAL_RANK"])
            
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
        nprocs = torch.cuda.device_count()
        args.batch_size = int(
            args.batch_size / nprocs)
    else:
        args.local_rank = 0
    print("local rank is set to: " + str(args.local_rank))
    return device

def get_logger(args):
    ## logger ##
    if not args.ddp and args.local_rank==0:
        set_logging_defaults(args.prefix,'log.txt')
        logger = logging.getLogger('main')
    elif args.ddp and args.local_rank==0: # distributed ddp
        if int(torch.__version__.split('.')[0])>=2 :
            #torch 2.0 with torchrun
            set_logging_defaults(args.prefix,'log.txt')
            logger = logging.getLogger('main')
        else:
            #torch 1.x with distributed
            logger = create_distributed_logging(args.prefix,'log.txt')
    return logger

def cleanup():
    import torch.distributed as dist
    dist.destroy_process_group()

if __name__ == '__main__':
    args = get_args(sys.argv[1:])

    ## gpu parallel ##
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    # set ddp
    
    # make results directory
    if not os.path.exists('./results'):
        os.makedirs('./results', exist_ok=True)
    device = set_ddp(args)

    # mp.spawn(main, args=(torch.cuda.device_count(),), nprocs=torch.cuda.device_count(), join=True)
    
    main(args)
    cleanup()
