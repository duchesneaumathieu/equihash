import os, sys, json, torch, argparse

from equihash.utils import timestamp
from equihash.utils.config import load_config, build_loader, build_network, build_trainer
from equihash.utils.states import save_state, load_state
from equihash.evaluation import QuickResults

def evaluate(trainer, net, train_loader, valid_loader, train_results, valid_results, eval_size):
    net.eval()
    if trainer.step == 0:
        print(timestamp(f'Initial evaluation...'), flush=True)
    else:
        trainer.aggregate()
        print(timestamp(trainer.training_log.describe(-1)), flush=True)
        
    train_results.evaluate(trainer.step, net, train_loader, batch_size=250, nb_documents=eval_size, seed=0xface)
    print(timestamp(train_results.describe(-1)), flush=True)
    
    valid_results.evaluate(trainer.step, net, valid_loader, batch_size=250, nb_documents=eval_size, seed=0xfade)
    print(timestamp(valid_results.describe(-1)), flush=True)

def main(task, model, variant, variant_id, load_checkpoint, checkpoints, force_load, device, stochastic, eval_first, no_save, args):
    config = load_config(task, model, variant=variant, variant_id=variant_id, args=args)
    #unpacking useful config item
    name = config['name']
    seed = config['seed']
    nb_steps = config['nb_steps']
    eval_size = config['eval_size']
    eval_freq = config['eval_freq']
    
    if not stochastic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(seed)
    
    save_folder = os.path.join('data', 'experiments', task, model, name)
    state_path = os.path.join(save_folder, 'current', 'state.pth')
    checkpoints_path = os.path.join(save_folder, 'checkpoint{step}', 'state.pth')
    
    train_loader = build_loader(config, which='train', device=device, seed=seed)
    valid_loader = build_loader(config, which='valid', device=device, seed=0xbeef)

    net = build_network(config, device=device)
    trainer = build_trainer(config, net, train_loader)
    train_results = QuickResults(which='train')
    valid_results = QuickResults(which='valid')
    
    #try to load existing models or creating a new one
    if load_checkpoint is not None:
        load_path = checkpoints_path.format(step=load_checkpoint)
        print(timestamp('Loading checkpoint...'), end='', flush=True)
        load_state(load_path, config, net, trainer, train_results, valid_results, force=force_load, verbose=True)
        print(f' (step={trainer.step})', flush=True)
    elif os.path.exists(state_path):
        print(timestamp('Loading models...'), end='', flush=True)
        load_state(state_path, config, net, trainer, train_results, valid_results, force=force_load, verbose=True)
        print(f' (step={trainer.step})', flush=True)
    else:
        print(timestamp('New model created.'), flush=True)
        save_state(state_path, config, net, trainer, train_results, valid_results)
    
    if trainer.step==0 and eval_first:
        evaluate(trainer, net, train_loader, valid_loader, train_results, valid_results, eval_size)

    if trainer.step==0 and 0 in checkpoints:
        print(timestamp(f'Creating checkpoint... (step={trainer.step})'), flush=True)
        save_path = checkpoints_path.format(step=trainer.step)
        save_state(save_path, config, net, trainer, train_results, valid_results)

    try:
        while trainer.step < nb_steps:
            net.train()
            nb_train_steps = eval_freq - trainer.step%eval_freq
            trainer.train(nb_steps=nb_train_steps)
            
            evaluate(trainer, net, train_loader, valid_loader, train_results, valid_results, eval_size)
            save_state(state_path, config, net, trainer, train_results, valid_results)
            if trainer.step in checkpoints and not no_save:
                print(timestamp(f'Creating checkpoint... (step={trainer.step})'), flush=True)
                save_path = checkpoints_path.format(step=trainer.step)
                save_state(save_path, config, net, trainer, train_results, valid_results)
    except KeyboardInterrupt: print('\nBye!')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('task', type=str, help='The experiment\'s task.')
    parser.add_argument('model', type=str, help='The experiment\'s model.')
    parser.add_argument('-v', '--variant', type=str, required=False, help='The model\'s variant.')
    parser.add_argument('-i', '--variant_id', type=int, required=False, help='The model\'s variant id.')
    parser.add_argument('-l', '--load_checkpoint', type=int, required=False, help='The step to resume from, the checkpoint must exist.')
    parser.add_argument('-p', '--checkpoints', nargs="*", type=int, required=False, default=list(),
                        help='A sequence of steps. At each of those step a checkpoints will be saved.')
    parser.add_argument('-f', '--force_load', action='store_true', default=False, help='Load the state even if the configs doesn\'t match.')
    parser.add_argument('-d', '--device', type=str, default='cuda', help='The device to use (cpu or cuda).')
    parser.add_argument('-s', '--stochastic', action='store_true',
                        help='If set, pytorch might use stochastic operations making the experiment not reproducible.')

    parser.add_argument('-e', '--eval_first', action='store_true', default=False,
                        help='If set, the model will be evaluated before the training starts.')
    parser.add_argument('--no_save', action='store_true', default=False, help='If set, the model won\'t be saved.')
    parser.add_argument(
        '-a', '--args', nargs="*", type=str,
        help="To overwrite config arguments. e.g. trainer_kwargs:alpha:0.15"
    )
    args = parser.parse_args()
    main(**vars(args))