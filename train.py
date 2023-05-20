import os, sys, json, torch, argparse

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(script_dir) #to access the equihash module
from equihash.utils import timestamp
from equihash.utils.config import build_loader, build_network, build_trainer
from equihash.utils.states import save_state, load_state
from equihash.evaluation import QuickResults

parser = argparse.ArgumentParser()
parser.add_argument('config', type=str, help='The path of the model\'s configuration json file.')
parser.add_argument('-m', '--model_name', type=str, required=False, default=None,
                    help='The model name, the default is the config name (without the extension).')
parser.add_argument('-s', '--states_dir', type=str, required=False, default=f'{script_dir}/states/',
                    help='The directory to save the running states and checkpoints. If set to /dev/null, the state won\'t be saved.')
parser.add_argument('-f', '--eval_freq', type=int, required=False, default=1_000, help='The evaluation frequency (in number of steps).')
parser.add_argument('-n', '--eval_size', type=int, required=False, default=100, help='The number of data in thousands in each evaluation.')
parser.add_argument('-l', '--load_checkpoint', type=int, required=False, help='The step to resume from, the checkpoint must exist.')
parser.add_argument('-p', '--checkpoints', nargs="*", type=int, required=False, default=list(),
                    help='A sequence of steps. At each of those step a checkpoints will be saved.')
parser.add_argument('-F', '--force_load', action='store_true', default=False, help='Load the state even if the configs doesn\'t match.')
parser.add_argument('-d', '--device', type=str, default='cuda', help='The device to use (cpu or cuda).')
parser.add_argument('-t', '--train_seed', type=int, default=0xcafe, help='The training seed.')
parser.add_argument('-D', '--deterministic', type=int, default=None,
                    help='Set the training seed, setting it makes the training deterministic.')
parser.add_argument('-e', '--eval_first', action='store_true', default=False,
                    help='If set, the model will be evaluated before the training starts.')
args = parser.parse_args()
args.eval_size *= 1000

if args.deterministic is not None:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.deterministic)

model_name, extension = os.path.splitext(os.path.basename(args.config))
model_name = model_name if args.model_name is None else args.model_name
state_path = os.path.join(args.states_dir, f'{model_name}.pth')
checkpoints_path = os.path.join(args.states_dir, f'{model_name}_step_{{step}}.pth')

with open(args.config, 'r') as f:
    config = json.load(f)

train_loader = build_loader(config, which='train', device=args.device, seed=args.deterministic)
valid_loader = build_loader(config, which='valid', device=args.device, seed=0xbeef)

net = build_network(config, device=args.device)
trainer = build_trainer(config, net, train_loader)
train_results = QuickResults(which='train')
valid_results = QuickResults(which='valid')

if args.load_checkpoint is not None:
    load_path = checkpoints_path.format(step=args.load_checkpoint)
    print(timestamp('Loading checkpoint...'), end='', flush=True)
    config = load_state(load_path, config, net, trainer, train_results, valid_results, force=args.force_load, verbose=True)
    print(f' (step={trainer.step})', flush=True)
elif os.path.exists(state_path):
    print(timestamp('Loading models...'), end='', flush=True)
    config = load_state(state_path, config, net, trainer, train_results, valid_results, force=args.force_load, verbose=True)
    print(f' (step={trainer.step})', flush=True)
else:
    print(timestamp('New model created.'), flush=True)
    save_state(state_path, config, net, trainer, train_results, valid_results)

if trainer.step==0 and args.eval_first:
    net.eval()
    train_results.evaluate(trainer.step, net, train_loader, batch_size=250, nb_documents=args.eval_size, seed=0xface)
    valid_results.evaluate(trainer.step, net, valid_loader, batch_size=250, nb_documents=args.eval_size, seed=0xfade)
    print(timestamp(f'Initial evaluation...'), flush=True)
    print(train_results.describe(-1), flush=True)
    print(valid_results.describe(-1), flush=True)

if trainer.step==0 and 0 in args.checkpoints:
    print(timestamp(f'Creating checkpoint... (step={trainer.step})'), flush=True)
    save_path = checkpoints_path.format(step=trainer.step)
    save_state(save_path, config, net, trainer, train_results, valid_results)

try:
    while trainer.step < 100_000:
        net.train()
        nb_train_steps = args.eval_freq - trainer.step%args.eval_freq
        trainer.train(nb_steps=nb_train_steps)
        
        net.eval()
        trainer.aggregate()
        train_results.evaluate(trainer.step, net, train_loader, batch_size=250, nb_documents=args.eval_size, seed=0xface)
        valid_results.evaluate(trainer.step, net, valid_loader, batch_size=250, nb_documents=args.eval_size, seed=0xfade)
        
        print(timestamp(trainer.training_log.describe(-1)), flush=True)
        print(train_results.describe(-1), flush=True)
        print(valid_results.describe(-1), flush=True)
        
        save_state(state_path, config, net, trainer, train_results, valid_results)
        if trainer.step in args.checkpoints:
            print(timestamp(f'Creating checkpoint... (step={trainer.step})'), flush=True)
            save_path = checkpoints_path.format(step=trainer.step)
            save_state(save_path, config, net, trainer, train_results, valid_results)
except KeyboardInterrupt: print('\nBye!')