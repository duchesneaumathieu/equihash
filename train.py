import os, sys, json, torch, argparse

from equihash.utils import timestamp
from equihash.utils.config import load_config, build_loader, build_network, build_trainer
from equihash.utils.states import save_state, load_state
from equihash.evaluation import QuickResults
from evaluate_binary_equihash import evaluate_binary_equihash

def evaluate(trainer, net, train_loader, valid_loader, train_results, valid_results, eval_size):
    net.eval()
    if trainer.step == 0:
        print(timestamp(f'Initial evaluation...'), flush=True)
    else:
        trainer.aggregate()
        print(timestamp(trainer.training_log.describe(trainer.step)), flush=True)
        
    train_results.evaluate(trainer.step)
    print(timestamp(train_results.describe(trainer.step)), flush=True)
    
    valid_results.evaluate(trainer.step)
    print(timestamp(valid_results.describe(trainer.step)), flush=True)

def main(task, model, variant, variant_id, load_checkpoint, checkpoints,
         force_load, device, stochastic, eval_first, no_save, database_name, args):
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
    train_results = QuickResults(net=net, loader=train_loader, nb_documents=eval_size, batch_size=250, seed=0xface)
    valid_results = QuickResults(net=net, loader=valid_loader, nb_documents=eval_size, batch_size=250, seed=0xfade)
    
    #try to load existing models or creating a new one
    if load_checkpoint is not None:
        load_path = checkpoints_path.format(step=load_checkpoint)
        print(timestamp(f'Loading checkpoint for {task}:{model}:{name}...'), end='', flush=True)
        load_state(load_path, config, net, trainer, train_results, valid_results, force=force_load, verbose=True)
        print(f' (step={trainer.step})', flush=True)
    elif os.path.exists(state_path):
        print(timestamp(f'Loading model {task}:{model}:{name}...'), end='', flush=True)
        load_state(state_path, config, net, trainer, train_results, valid_results, force=force_load, verbose=True)
        print(f' (step={trainer.step})', flush=True)
    else:
        print(timestamp(f'New model created - {task}:{model}:{name}'), flush=True)
        if not no_save: save_state(state_path, config, net, trainer, train_results, valid_results)
    
    if trainer.step==0 and eval_first:
        evaluate(trainer, net, train_loader, valid_loader, train_results, valid_results, eval_size)

    if not no_save and trainer.step==0 and 0 in checkpoints:
        print(timestamp(f'Creating checkpoint... (step={trainer.step})'), flush=True)
        save_path = checkpoints_path.format(step=trainer.step)
        save_state(save_path, config, net, trainer, train_results, valid_results)

    try:
        while trainer.step < nb_steps:
            net.train()
            nb_train_steps = eval_freq - trainer.step%eval_freq
            trainer.train(nb_steps=nb_train_steps)
            
            evaluate(trainer, net, train_loader, valid_loader, train_results, valid_results, eval_size)
            if not no_save: save_state(state_path, config, net, trainer, train_results, valid_results)
            if not no_save and trainer.step in checkpoints:
                print(timestamp(f'Creating checkpoint... (step={trainer.step})'), flush=True)
                save_path = checkpoints_path.format(step=trainer.step)
                save_state(save_path, config, net, trainer, train_results, valid_results)
    except KeyboardInterrupt:
        print()
        print(timestamp('Bye!'))
        sys.exit(0)
        
    if database_name is not None:
        print() #newline
        evaluate_binary_equihash(task, database_name, model, variant, variant_id,
                                 load_checkpoint=None, which='test', device=device, encode=True, verbose=True)
    
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
        '--database_name', type=str, required=False, default=None,
        help='The database name used for testing after the training is complete. If nothing is provided, no test will be performed.'
    )
    parser.add_argument(
        '-a', '--args', nargs="*", type=str,
        help="To overwrite config arguments. e.g. trainer_kwargs:alpha:0.15"
    )
    args = parser.parse_args()
    main(**vars(args))