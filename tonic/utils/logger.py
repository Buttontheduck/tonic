import datetime
import os
import time
import torch
import numpy as np
import termcolor
import yaml
import wandb

current_logger = None


class Logger:
    '''Logger used to display and save logs, and save experiment configs.'''

    def __init__(self, path=None, width=60, script_path=None, config=None):
        self.path = path or str(time.time())
        self.log_file_path = os.path.join(self.path, 'log.csv')

        # Save the launch script.
        if script_path:
            with open(script_path, 'r') as script_file:
                script = script_file.read()
                try:
                    os.makedirs(self.path, exist_ok=True)
                except Exception:
                    pass
                script_path = os.path.join(self.path, 'script.py')
                with open(script_path, 'w') as config_file:
                    config_file.write(script)
                log(f'Script file saved to {script_path}')

        # Save the configuration.
        if config:
            try:
                os.makedirs(self.path, exist_ok=True)
            except Exception:
                pass
            config_path = os.path.join(self.path, 'config.yaml')
            with open(config_path, 'w') as config_file:
                yaml.dump(config, config_file)
            log(f'Config file saved to {config_path}')

        self.known_keys = set()
        self.stat_keys = set()
        self.w_keys = set()
        self.epoch_dict = {}
        self.width = width
        self.last_epoch_progress = None
        self.start_time = time.time()

    def store(self, key, value, stats=False, log_weights=False):
        '''Keeps named values during an epoch.'''

        if key not in self.epoch_dict:
            self.epoch_dict[key] = [value]
            
            if log_weights:
                self.w_keys.add(key)       
                stats = False    
                
            if stats:
                self.stat_keys.add(key)

        else:
            self.epoch_dict[key].append(value)

    def dump(self):
        '''Displays and saves the values at the end of an epoch.'''

        # Compute statistics if needed.
        keys = list(self.epoch_dict.keys())
        for key in keys:
            print(key)
            values = self.epoch_dict[key]
            
            if key in self.stat_keys:
                self.epoch_dict[key + '/mean'] = np.mean(values)
                self.epoch_dict[key + '/std'] = np.std(values)
                self.epoch_dict[key + '/min'] = np.min(values)
                self.epoch_dict[key + '/max'] = np.max(values)
                self.epoch_dict[key + '/size'] = len(values)
                del self.epoch_dict[key]
                
            if key in self.w_keys:
                """
                For weight values, tensor have the size [TS,S,BS] 
                - TS: # of timesteps of interaction with the environment, in our case 1000
                - S:  # of actions sampled given the state, in our case 20
                - BS: # of states in a batch, batch size, in our case 256
                
                + so we end up val tensor, torch.Size([1000, 20, 256])
                """

                val = torch.stack(values)
                
                val_100 = val[100]
                val_199 = val[199]
                val_500 = val[500]
                val_599 = val[599]     
                val_900 = val[900]
                val_999 = val[999]    
                
                wandb.log({key+'/Weights Per Time/ T=100': wandb.Histogram(val_100.numpy())})
                wandb.log({key+'/Weights Per Time/ T=199': wandb.Histogram(val_199.numpy())})                
                wandb.log({key+'/Weights Per Time/ T=500': wandb.Histogram(val_500.numpy())})
                wandb.log({key+'/Weights Per Time/ T=599': wandb.Histogram(val_599.numpy())})                
                wandb.log({key+'/Weights Per Time/ T=900': wandb.Histogram(val_900.numpy())})
                wandb.log({key+'/Weights Per Time/ T=999': wandb.Histogram(val_999.numpy())})                
                                                                                       
                                        
                max_mean_vals = torch.mean(val.amax(dim=(1)))
                min_mean_vals = torch.mean(val.amin(dim=(1)))
                max_max_vals = val.amax(dim=(1, 2))  # for each timestep TS, max weight (W) of each (s,a) weights. [1000]
                min_min_vals = val.amin(dim=(1, 2))  # for each timestep TS, min weight (W) of each (s,a) weights. [1000]

                mean = torch.mean(val)
                max2_mean = torch.mean(max_max_vals)                      
                min2_mean = torch.mean(min_min_vals)   

                self.epoch_dict[key+'/Mean/max_max_mean']  = max2_mean.numpy()
                self.epoch_dict[key+'/Mean/min_min_mean']  = min2_mean.numpy()
                self.epoch_dict[key+'/Mean/max_mean_mean'] = max_mean_vals.numpy()
                self.epoch_dict[key+'/Mean/min_mean_mean'] = min_mean_vals.numpy()
                
                self.epoch_dict[key+'/Dist/max_max_dist'] = max_max_vals.numpy()
                self.epoch_dict[key+'/Dist/min_min_dist'] = min_min_vals.numpy()
                
                
                self.epoch_dict[key+'/mean'] = mean.numpy()
                
                

            else:
                self.epoch_dict[key] = np.mean(values)

        # Check if new keys were added.
        new_keys = [key for key in self.epoch_dict.keys()
                    if key not in self.known_keys]
        if new_keys:
            first_row = len(self.known_keys) == 0
            if not first_row:
                print()
                warning(f'Logging new keys {new_keys}')
            # List the keys and prepare the display layout.
            for key in new_keys:
                self.known_keys.add(key)
            self.final_keys = list(sorted(self.known_keys))
            self.console_formats = []
            known_keys = set()
            for key in self.final_keys:
                print(key)
                *left_keys, right_key = key.split('/')
                for i, k in enumerate(left_keys):
                    left_key = '/'.join(left_keys[:i + 1])
                    if left_key not in known_keys:
                        left = '  ' * i + k.replace('_', ' ')
                        self.console_formats.append((left, None))
                        known_keys.add(left_key)
                indent = '  ' * len(left_keys)
                right_key = right_key.replace('_', ' ')
                self.console_formats.append((indent + right_key, key))

        # Display the values following the layout.
        print()
        for left, key in self.console_formats:
            print(key)
            if key:
                val = self.epoch_dict.get(key)
                str_type = str(type(val))
                if 'tensorflow' in str_type:
                    warning(f'Logging TensorFlow tensor {key}')
                elif 'torch' in str_type:
                    warning(f'Logging Torch tensor {key}')
                if np.issubdtype(type(val), np.floating):
                    right = f'{val:8.3g}'
                elif np.issubdtype(type(val), np.integer):
                    right = f'{val:,}'
                else:
                    right = str(val)
                spaces = ' ' * (self.width - len(left) - len(right))
                print(left + spaces + right)
            else:
                spaces = ' ' * (self.width - len(left))
                print(left + spaces)
        print()

        # Save the data to the log file
        vals = [self.epoch_dict.get(key) for key in self.final_keys]
        if new_keys:
            if first_row:
                log(f'Logging data to {self.log_file_path}')
                try:
                    os.makedirs(self.path, exist_ok=True)
                except Exception:
                    pass
                with open(self.log_file_path, 'w') as file:
                    file.write(','.join(self.final_keys) + '\n')
                    file.write(','.join(map(str, vals)) + '\n')
            else:
                with open(self.log_file_path, 'r') as file:
                    lines = file.read().splitlines()
                old_keys = lines[0].split(',')
                old_lines = [line.split(',') for line in lines[1:]]
                new_indices = []
                j = 0
                for i, key in enumerate(self.final_keys):
                    if key == old_keys[j]:
                        j += 1
                    else:
                        new_indices.append(i)
                assert j == len(old_keys)
                for line in old_lines:
                    for i in new_indices:
                        line.insert(i, 'None')
                with open(self.log_file_path, 'w') as file:
                    file.write(','.join(self.final_keys) + '\n')
                    for line in old_lines:
                        file.write(','.join(line) + '\n')
                    file.write(','.join(map(str, vals)) + '\n')
        else:
            with open(self.log_file_path, 'a') as file:
                file.write(','.join(map(str, vals)) + '\n')

        wandb.log(self.epoch_dict)
        self.epoch_dict.clear()
        self.last_epoch_progress = None
        self.last_epoch_time = time.time()

    def show_progress(
        self, steps, num_epoch_steps, num_steps, color='white',
        on_color='on_blue'
    ):
        '''Shows a progress bar for the current epoch and total training.'''

        epoch_steps = (steps - 1) % num_epoch_steps + 1
        epoch_progress = int(self.width * epoch_steps / num_epoch_steps)
        if epoch_progress != self.last_epoch_progress:
            current_time = time.time()
            seconds = current_time - self.start_time
            seconds_per_step = seconds / steps
            epoch_rem_steps = num_epoch_steps - epoch_steps
            epoch_rem_secs = max(epoch_rem_steps * seconds_per_step, 0)
            epoch_rem_secs = datetime.timedelta(seconds=epoch_rem_secs + 1e-6)
            epoch_rem_secs = str(epoch_rem_secs)[:-7]
            total_rem_steps = num_steps - steps
            total_rem_secs = max(total_rem_steps * seconds_per_step, 0)
            total_rem_secs = datetime.timedelta(seconds=total_rem_secs)
            total_rem_secs = str(total_rem_secs)[:-7]
            msg = f'Time left:  epoch {epoch_rem_secs}  total {total_rem_secs}'
            msg = msg.center(self.width)
            print(termcolor.colored(
                '\r' + msg[:epoch_progress], color, on_color), end='')
            print(msg[epoch_progress:], sep='', end='')
            self.last_epoch_progress = epoch_progress


def initialize(*args, **kwargs):
    global current_logger
    current_logger = Logger(*args, **kwargs)
    return current_logger


def get_current_logger():
    global current_logger
    if current_logger is None:
        current_logger = Logger()
    return current_logger


def store(*args, **kwargs):
    logger = get_current_logger()
    return logger.store(*args, **kwargs)


def dump(*args, **kwargs):
    logger = get_current_logger()
    return logger.dump(*args, **kwargs)


def show_progress(*args, **kwargs):
    logger = get_current_logger()
    return logger.show_progress(*args, **kwargs)


def get_path():
    logger = get_current_logger()
    return logger.path


def log(msg, color='green'):
    print(termcolor.colored(msg, color, attrs=['bold']))


def warning(msg, color='yellow'):
    print(termcolor.colored('Warning: ' + msg, color, attrs=['bold']))


def error(msg, color='red'):
    print(termcolor.colored('Error: ' + msg, color, attrs=['bold']))