"""
Code to get files from other servers. It saves the file in exactly same location. It also helps to look up all log files
in last one day.
"""
import argparse
import logging
import os
import pipes
import socket
import subprocess

SERVER_LIST = [
    'cl4.learner.csie.ntu.edu.tw',
    'cl5.learner.csie.ntu.edu.tw',
    'cl6.learner.csie.ntu.edu.tw',
    'cl7.learner.csie.ntu.edu.tw',
    'cl8.learner.csie.ntu.edu.tw',
]
CHECKPOINT_DIR = '/tmp2/ashesh/checkpoints/'
LOG_DIR = '/tmp2/ashesh/logs/'

SERVER = socket.gethostname()
logging.basicConfig(
    format='[%(server)s %(asctime)s ] %(levelname)s - %(message)s', datefmt='%Y/%m/%d %H:%M:%S', level=logging.INFO)

logger = logging.getLogger(__name__)
logger = logging.LoggerAdapter(logger, {'server': SERVER})


def human_readable_size(sz):
    sz = int(sz)
    if sz / 10**9 > 1:
        return f'{round(sz/1024**3)}G'
    elif sz / 10**6 > 1:
        return f'{round(sz/1024**2)}M'
    elif sz / 10**3 > 1:
        return f'{round(sz/1024)}K'
    return str(sz)


def recent_files(directory, extension):
    cmds = ['find', directory, '-maxdepth', '1', '-mtime', '-1', '-ls']
    myCommandStr = ' '.join([pipes.quote(n) for n in cmds])
    output_dict = {}
    for server_ip in SERVER_LIST:
        p = subprocess.Popen(['ssh', '-t', server_ip, myCommandStr], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        data = p.communicate()
        lines = data[0].decode('utf-8').splitlines()
        lines = [l for l in lines if l[-len(extension):] == extension]
        # Example line
        # ['38894549', '46996', '-rw-r--r--', '1', 'ashesh', 'cllab', '48120582', 'Apr', '25', '14:20',
        #  '/tmp2/ashesh/checkpoints/gaze360_static_TYPE:22_fc2:256_dpred_weight:0.01.pth.tar']
        if len(lines) == 0:
            continue
        token_set = [l.split()[6:] for l in lines]
        clean_lines = []
        for tokens in token_set:
            tokens[0] = human_readable_size(tokens[0])
            clean_lines.append(' '.join(tokens))

        output_dict[server_ip] = clean_lines

    for server_ip in output_dict:
        print(server_ip)
        print('\n'.join(output_dict[server_ip]))


def recent_logs():
    recent_files(LOG_DIR, 'txt')


def recent_checkpoints():
    recent_files(CHECKPOINT_DIR, '.pth.tar')


def check_existance(fpath):
    """
    It must be a file. If it is a directory, it will not work.
    """
    relv_servers = []
    for server_ip in SERVER_LIST:
        cmd = ['ssh', server_ip, 'test', '-f', fpath]
        p = subprocess.Popen(cmd)
        _ = p.communicate()
        if p.returncode == 0:
            relv_servers.append(server_ip)
    logger.info(f'{fpath} is present on {[s.split(".")[0] for s in relv_servers]}')
    return relv_servers


def fetch_from_server(server_ip, fpath, target_fpath=None):
    if target_fpath is None:
        target_fpath = fpath

    assert os.path.exists(fpath) is False, f'File:{fpath} is present on {SERVER}!'

    p = subprocess.Popen(["scp", f"{server_ip}:{fpath}", fpath])
    _ = os.waitpid(p.pid, 0)
    if os.path.exists(fpath):
        return 1
    return 0


def fetch(fpath):
    if os.path.exists(fpath):
        logger.info(f'{fpath} already exists.')
        return 1
    relv_server_list = check_existance(fpath)
    if len(relv_server_list) == 1:
        server_ip = relv_server_list[0]
        server_name = server_ip.split('.')[0]
        logger.info(f'Trying to fetch from {server_name}')
        if fetch_from_server(server_ip, fpath):
            logger.info(f'{fpath} fetched from "{server_ip.split(".")[0]}".')
            return 1
        else:
            logging.error("Not able to fetch when the file exists on that server. Aborting")
            return 0

    elif len(relv_server_list) == 0:
        logger.warning(f'{fpath} not found anywhere !!')
        return 0
    else:
        logger.warning('Multiple instances exists. Please clean relevant ones first. Aborting !')
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--recent_logs', action='store_true')
    parser.add_argument('--recent_checkpoints', action='store_true')
    parser.add_argument(
        '--fpath', type=str, default='', help='Full path of the file. On all servers, this fpath will be checked')
    parser.add_argument('--check', action='store_true', help='Check on which servers do the file exists')
    args = parser.parse_args()
    if args.recent_logs:
        recent_logs()
    elif args.recent_checkpoints:
        recent_checkpoints()
    elif args.check:
        check_existance(args.fpath)
    else:
        fetch(args.fpath)
