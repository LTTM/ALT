import json
import torch.optim as optim
import argparse
import copy
import datetime
import wandb
import random

from utils import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet50',
                        choices=["resnet50", "cnn"],
                        help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset used for training')
    parser.add_argument('--partition', type=str, default='homo', help='the data partitioning strategy')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate (default: 0.1)')
    parser.add_argument('--epochs', type=int, default=5, help='number of local epochs')
    parser.add_argument('--n_clients', type=int, default=2, help='number of workers in a distributed cluster')
    parser.add_argument('--alg', type=str, default='ALT')
    parser.add_argument('--N_rounds', type=int, default=50, help='number of maximum communication rounds')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--modeldir', type=str, required=False, default="./saved_models/", help='Model directory path')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    parser.add_argument('--local_max_epoch', type=int, default=100,
                        help='the number of epoch for local optimal training')
    parser.add_argument('--model_buffer_size', type=int, default=1,
                        help='store how many previous models for contrastive loss')
    parser.add_argument('--pool_option', type=str, default='FIFO', help='FIFO or BOX')
    parser.add_argument('--sample_fraction', type=float, default=1.0, help='how many clients are sampled in each round')
    parser.add_argument('--load_model_file', type=str, default=None, help='the model to load as global model')
    parser.add_argument('--load_model_round', type=int, default=None,
                        help='how many rounds have executed for the loaded model')
    parser.add_argument('--normal_model', type=int, default=0, help='use normal model or aggregate model')
    parser.add_argument('--save_model', type=int, default=0)
    parser.add_argument('--use_project_head', type=int, default=1)
    parser.add_argument('--server_momentum', type=float, default=0, help='the server momentum (FedAvgM)')
    parser.add_argument('--lambda_ce', type=float, default=1)
    parser.add_argument('--lambda_moon', type=float, default=0)
    parser.add_argument('--dynamic_epochs', type=int, default=0,
                        help="when true allow clients to train for a variable number of epochs")
    parser.add_argument('--threshold_sim', type=str, default="0.8",
                        help="0.1, 0.3, 0.5, 0.7, 0.9, linear")
    parser.add_argument('--threshold_start', type=float, default="0.1")
    parser.add_argument('--threshold_end', type=float, default="0.9")
    parser.add_argument('--th_descriptor', type=str, default="mean",
                        help="other examples: max1, max2, min3, min, sample ecc. it is used on the sim before the th")
    parser.add_argument('--wandb', type=str, default="online")
    parser.add_argument('--name', type=str, default="")
    args = parser.parse_args()
    return args


def init_nets(n_clients, args, device='cpu'):
    nets = {net_i: None for net_i in range(n_clients)}
    if args.dataset == 'cifar10':
        n_classes = 10
    elif args.dataset == 'tinyimagenet':
        n_classes = 200
    if args.normal_model:
        for net_i in range(n_clients):
            if args.model == 'cnn':
                net = SimpleCNNMNIST(input_dim=(16 * 4 * 4), hidden_dims=[120, 84], output_dim=10)
            if device == 'cpu':
                net.to(device)
            else:
                net = net.cuda()
            nets[net_i] = net
    else:
        for net_i in range(n_clients):
            if args.use_project_head:
                net = ModelFed(args.model, args.out_dim, n_classes)
            else:
                net = ModelFed_noheader(args.model, args.out_dim, n_classes)
            if device == 'cpu':
                net.to(device)
            else:
                net = net.cuda()
            nets[net_i] = net

    model_meta_data = []
    layer_type = []
    for (k, v) in nets[0].state_dict().items():
        model_meta_data.append(v.shape)
        layer_type.append(k)

    return nets, model_meta_data, layer_type



def train_net_ALT(net_id, net, global_net, previous_nets, train_dataloader, test_dataloader, epochs, lr,
                      args_optimizer, temperature, args,
                      round, device="cpu"):
    net = nn.DataParallel(net)
    net.cuda()
    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    train_acc, _ = compute_accuracy(net, train_dataloader, args=args, device=device)

    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True,
                                                args=args, device=device)

    logger.info('>> Pre-Training Training accuracy: {}'.format(train_acc))
    logger.info('>> Pre-Training Test accuracy: {}'.format(test_acc))


    if args_optimizer == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg)
    elif args_optimizer == 'amsgrad':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, weight_decay=args.reg,
                               amsgrad=True)
    elif args_optimizer == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr, momentum=0.9,
                              weight_decay=args.reg)

    criterion = nn.CrossEntropyLoss().cuda()

    if args.th_descriptor == "sample":
        th_descriptor = None
    else:
        th_descriptor = ThresholdDescriptor(name=args.th_descriptor)

    for previous_net in previous_nets:
        previous_net.cuda()

    cnt = 0
    cos = torch.nn.CosineSimilarity(dim=-1)
    epoch = 0


    if args.threshold_sim == "linear":
        slope = args.threshold_end - args.threshold_start
        threshold_sim = args.threshold_start + (round * slope) / (args.N_rounds - 1)

    else:
        threshold_sim = float(args.threshold_sim)

    if args.dynamic_epochs:
        wandb_logger.log({f"threshold": threshold_sim, "round": round})

    for epoch in range(epochs):
        early_stopping = 0

        epoch_loss_collector = []

        epoch_loss_collection_partial = {
            "ce": [],
            "moon": []
        }

        for batch_idx, (x, target) in enumerate(train_dataloader):

            x, target = x.cuda(), target.cuda()

            optimizer.zero_grad()
            x.requires_grad = False
            target.requires_grad = False
            target = target.long()

            loss = {
                "ce": 0,
                "moon": 0,
            }

            if args.lambda_moon or args.lambda_ce or args.dynamic_epochs:
                _, pro1, out = net(x)
                _, pro2, _ = global_net(x)

                posi = cos(pro1, pro2)
                logits = posi.reshape(-1, 1)

                if args.th_descriptor == "sample" and len(torch.where(posi < threshold_sim)[0]):
                    indexes = torch.where(posi >= threshold_sim)
                    # stop the training if more than half samples should be discarded
                    if len(indexes[0]) < int(target.shape[0]/2):
                        early_stopping = 1
                        break
                    x = x[indexes]
                    logits = logits[indexes]
                    out = out[indexes]
                    x_ind = torch.cat((indexes[0], indexes[0]+labels.size(0)), dim=0)
                    x = x[x_ind]
                    pro1 = pro1[indexes]
                    target = target[indexes]

                elif th_descriptor and th_descriptor.compute(posi) < threshold_sim and args.dynamic_epochs:
                    early_stopping = 1

                for previous_net in previous_nets:
                    previous_net.cuda()
                    _, pro3, _ = previous_net(x)
                    nega = cos(pro1, pro3)
                    logits = torch.cat((logits, nega.reshape(-1, 1)), dim=1)

                    previous_net.to('cpu')

                if args.lambda_moon:
                    logits /= temperature
                    labels = torch.zeros(x.size(0)).cuda().long()
                    loss["moon"] = criterion(logits, labels)

                loss["ce"] = criterion(out, target)


            loss_tot = loss["moon"] * args.lambda_moon + \
                       loss["ce"] * args.lambda_ce

            loss_tot.backward()
            optimizer.step()

            cnt += 1
            epoch_loss_collector.append(loss_tot.item())

            for k in loss.keys():
                if loss[k]:
                    epoch_loss_collection_partial[k].append(loss[k].item())
                    wandb_logger.log({f"loss_{k}": loss[k], "round": round})

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Loss: %f' % (epoch, epoch_loss))

        if early_stopping:
            print(f"early stopped at epoch {epoch}")
            break

        for k, v in epoch_loss_collection_partial.items():
            if v:
                logger.info(f'Epoch: {epoch}  {k} Loss: {sum(v) / len(epoch_loss_collection_partial[k])}')

    for previous_net in previous_nets:
        previous_net.to('cpu')
    train_acc, _ = compute_accuracy(net, train_dataloader, args=args,  device=device)
    test_acc, conf_matrix, _ = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, args=args,
                                                device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)
    net.to('cpu')
    logger.info(' ** Training complete **')

    wandb_logger.log({f"epochs/client_{net_id}": epoch+1, "round": round})


    return train_acc, test_acc, epoch+1


def local_train_net(nets, args, net_dataidx_map, train_dl=None, test_dl=None, global_model=None, prev_model_pool=None,
                    server_c=None, clients_c=None, round=None, device="cpu"):
    avg_acc = 0.0
    acc_list = []
    if global_model:
        global_model.cuda()
    if server_c:
        server_c.cuda()
        server_c_collector = list(server_c.cuda().parameters())
        new_server_c_collector = copy.deepcopy(server_c_collector)

    epoch_list = []
    sizes_label = []

    for net_id, net in nets.items():
        dataidxs = net_dataidx_map[net_id]

        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        train_dl_local, test_dl_local, _, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs)

        n_epoch = args.epochs

        prev_models = []

        size_label = get_size_label(train_dl_local, n_classes)
        sizes_label.append(size_label)

        for i in range(len(prev_model_pool)):
            prev_models.append(prev_model_pool[i][net_id])
        trainacc, testacc, epochs = train_net_ALT(net_id, net, global_model, prev_models, train_dl_local, test_dl,
                                                n_epoch, args.lr,
                                                args.optimizer, args.temperature, args, round,
                                                device=device)
        epoch_list.append(epochs)


        logger.info("net %d final test acc %f" % (net_id, testacc))
        avg_acc += testacc
        acc_list.append(testacc)
    avg_acc /= args.n_clients
    if global_model:
        global_model.to('cpu')
    if server_c:
        for param_index, param in enumerate(server_c.parameters()):
            server_c_collector[param_index] = new_server_c_collector[param_index]
        server_c.to('cpu')

    return nets, epoch_list


if __name__ == '__main__':
    args = get_args()
    if args.wandb == "offline":
        os.environ["WANDB_MODE"] = "offline"
    wandb.login()

    rname = args.name
    rname += "_N" + str(args.n_clients)
    rname += "_alpha" + str(args.alpha)
    rname += "_r" + str(args.N_rounds)
    rname += "_bs" + str(args.batch_size)
    rname += "_e" + str(args.epochs)
    rname += "_lr" + str(args.lr)


    if args.lambda_ce:
        rname += "_Ce"

    if args.dynamic_epochs:
        rname += "_dyn_TH" + str(args.threshold_sim)

    wandb_logger = wandb.init(
        project="ALT",
        entity="",
        group=args.dataset,
        name=rname,
        config=vars(args)
    )

    mkdirs(args.logdir)
    mkdirs(args.modeldir)
    if args.log_file_name is None:
        argument_path = 'experiment_arguments-%s.json' % datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
    else:
        argument_path = args.log_file_name + '.json'
    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args), f)
    device = torch.device(args.device)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if args.log_file_name is None:
        args.log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S"))
    log_path = args.log_file_name + '.log'
    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.DEBUG, filemode='w')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.info(device)

    seed = args.init_seed
    logger.info("#" * 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)

    logger.info("Partitioning data")

    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        args.dataset, args.datadir, args.logdir, args.partition, args.n_clients, alpha=args.alpha)

    n_party_per_round = int(args.n_clients * args.sample_fraction)
    party_list = [i for i in range(args.n_clients)]
    party_list_rounds = []
    if n_party_per_round != args.n_clients:
        for i in range(args.N_rounds):
            party_list_rounds.append(random.sample(party_list, n_party_per_round))
    else:
        for i in range(args.N_rounds):
            party_list_rounds.append(party_list)

    n_classes = len(np.unique(y_train))

    train_dl_global, test_dl, train_ds_global, test_ds_global = get_dataloader(args.dataset,
                                                                               args.datadir,
                                                                               args.batch_size,
                                                                               32)

    print("len train_dl_global:", len(train_ds_global))
    train_dl = None
    data_size = len(test_ds_global)

    logger.info("Initializing nets")
    nets, local_model_meta_data, layer_type = init_nets(args.n_clients, args, device='cpu')

    global_models, global_model_meta_data, global_layer_type = init_nets(1, args, device='cpu')
    global_model = global_models[0]
    N_rounds = args.N_rounds
    if args.load_model_file:
        global_model.load_state_dict(torch.load(args.load_model_file))
        N_rounds -= args.load_model_round

    if args.server_momentum:
        moment_v = copy.deepcopy(global_model.state_dict())
        for key in moment_v:
            moment_v[key] = 0

    old_nets_pool = []

    for round in range(N_rounds):
        print("round:" + str(round))

        compute_cost_current_round = []

        logger.info("in comm round:" + str(round))
        party_list_this_round = party_list_rounds[round]
        nets_this_round = {k: nets[k] for k in party_list_this_round}

        global_model.eval()

        global_w = global_model.state_dict()

        if args.server_momentum:
            old_w = copy.deepcopy(global_model.state_dict())

        for net in nets_this_round.values():
            net.load_state_dict(global_w)

        _, epoch_list, sizes_label = local_train_net(nets_this_round, args, net_dataidx_map, train_dl=train_dl,
                                            test_dl=test_dl, global_model=global_model, prev_model_pool=old_nets_pool,
                                            round=round, device=device)

        total_data_points = sum([len(net_dataidx_map[r]) for r in party_list_this_round])
        fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in party_list_this_round]

        for net_id, net in enumerate(nets_this_round.values()):
            net_para = net.state_dict()
            if net_id == 0:
                for key in net_para:
                    global_w[key] = net_para[key] * fed_avg_freqs[net_id]
            else:
                for key in net_para:
                    global_w[key] += net_para[key] * fed_avg_freqs[net_id]

        if args.server_momentum:
            delta_w = copy.deepcopy(global_w)
            for key in delta_w:
                delta_w[key] = old_w[key] - global_w[key]
                moment_v[key] = args.server_momentum * moment_v[key] + (1 - args.server_momentum) * delta_w[key]
                global_w[key] = old_w[key] - moment_v[key]

        global_model.load_state_dict(global_w)


        logger.info('global n_training: %d' % len(train_dl_global))
        logger.info('global n_test: %d' % len(test_dl))
        global_model.cuda()
        train_acc, train_loss = compute_accuracy(global_model, train_dl_global, args=args, device=device)
        test_acc, conf_matrix, _ = compute_accuracy(global_model, test_dl, get_confusion_matrix=True, args=args,
                                                    device=device)
        global_model.to('cpu')
        logger.info('>> Global Model Train accuracy: %f' % train_acc)
        logger.info('>> Global Model Test accuracy: %f' % test_acc)
        logger.info('>> Global Model Train loss: %f' % train_loss)
        wandb_logger.log({"train_acc": train_acc, "round": round})
        wandb_logger.log({"test_acc": test_acc, "round": round})
        wandb_logger.log({"train_loss": train_loss, "round": round})


        logger.info(f">> Computation cost round {round} == {np.sum(epoch_list)}")
        print(f">> Computation cost round {round} == {np.sum(epoch_list)}")
        wandb_logger.log({"sum_epochs": np.sum(epoch_list), "round": round})
        wandb_logger.log({"min_epochs": np.min(epoch_list), "round": round})
        wandb_logger.log({"max_epochs": np.max(epoch_list), "round": round})



        if len(old_nets_pool) < args.model_buffer_size:
            old_nets = copy.deepcopy(nets)
            for _, net in old_nets.items():
                net.eval()
                for param in net.parameters():
                    param.requires_grad = False
            old_nets_pool.append(old_nets)
        elif args.pool_option == 'FIFO':
            old_nets = copy.deepcopy(nets)
            for _, net in old_nets.items():
                net.eval()
                for param in net.parameters():
                    param.requires_grad = False
            for i in range(args.model_buffer_size - 2, -1, -1):
                old_nets_pool[i] = old_nets_pool[i + 1]
            old_nets_pool[args.model_buffer_size - 1] = old_nets

        mkdirs(args.modeldir + 'ALT/')
        if args.save_model:
            torch.save(global_model.state_dict(),
                        args.modeldir + 'ALT/global_model_' + args.log_file_name + '.pth')
            torch.save(nets[0].state_dict(), args.modeldir + 'ALT/localmodel0' + args.log_file_name + '.pth')
            for nets_id, old_nets in enumerate(old_nets_pool):
                torch.save({'pool' + str(nets_id) + '_' + 'net' + str(net_id): net.state_dict() for net_id, net in
                            old_nets.items()},
                            args.modeldir + 'ALT/prev_model_pool_' + args.log_file_name + '.pth')

