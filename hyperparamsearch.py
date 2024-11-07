import torch
from Architectures.UnetCustom import UNet as UNetCustom
#from Architectures.UnetCustom import UNetPerf
from Architectures.UNetPerf import UNet as UNetPerf
from benchmarking import benchmark, get_datasets
from perforateCustomNet import perforate_net_perfconv, add_functs
weight_decay = 0.0005
epochs = 100
device = "cuda:0" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    batch_size = 32
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.1, 0.9])).to(device)
    net = UNetPerf(2, perforation_mode=(2,2))
    #net = UNet(2)
    #add_functs(net)
    #perforate_net_perfconv(net, in_size=(2,3,128,128))
    #net._set_perforation((2,2))
    op = torch.optim.SGD(net.parameters(), momentum=0.9, lr=0.00001, weight_decay=weight_decay)
    #op = torch.optim.Adam(net.parameters(), lr=0.0001, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(op, T_max=epochs)

    scheduler = torch.optim.lr_scheduler.SequentialLR(op, [torch.optim.lr_scheduler.ConstantLR(op, factor=0.00001), scheduler], milestones=[10])
    train_loader, valid_loader, test_loader = get_datasets("agri", batch_size, image_resolution=(128,128))
    perf = (2,2)
    run_name = "teeeeeeeeeeest"
    prefix = "h"
    eval_modes = [(1,1), (2,2), (3,3)]
    eval_modes = [None]
    in_size = (batch_size, 3, 128,128)
    perf_type = "perf"
    dataset = "agri"
    f = None
    net.to(device)
    bbest_out, confs, metrics = benchmark(net, op, scheduler, train_loader=train_loader,
                                                            valid_loader=valid_loader, test_loader=test_loader,
                                                            max_epochs=epochs, device=device, perforation_mode=perf,
                                                            run_name=run_name, batch_size=batch_size,
                                                            loss_function=loss_fn,prefix=prefix,
                                                            eval_modes=eval_modes, in_size=in_size, dataset=dataset,
                                                            perforation_type=perf_type, file=f, summarise=False,
                                                            grad_clip=None)
    #TODO !!!!!!!
    #try learning networks with non-strided backward! maybe that is the better effect