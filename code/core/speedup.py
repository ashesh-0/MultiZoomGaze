# Taken from https://medium.com/@data.scientist/speeding-up-pytorch-training-by-10-34508a85d6e6
import multiprocessing
import time

import torch


def loading_time(img_loader, num_workers, pin_memory, batch_size):
    kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory}
    train_loader = torch.utils.data.DataLoader(img_loader, batch_size=batch_size, shuffle=True, **kwargs)
    start = time.time()
    for epoch in range(4):
        for batch_idx, _ in enumerate(train_loader):
            if batch_idx == 15:
                break
            pass
    end = time.time()
    print("  Used {} second with num_workers = {}".format(end - start, num_workers))
    return end - start


def grid_search(img_loader, batch_size=64):
    core_number = multiprocessing.cpu_count()
    best_num_worker = [0, 0]
    best_time = [99999999, 99999999]
    print('cpu_count =', core_number)
    for pin_memory in [False, True]:
        print("While pin_memory =", pin_memory)
        for num_workers in range(0, core_number * 2 + 1, 4):
            current_time = loading_time(img_loader, num_workers, pin_memory, batch_size)
            if current_time < best_time[pin_memory]:
                best_time[pin_memory] = current_time
                best_num_worker[pin_memory] = num_workers
            else:  # assuming its a convex function
                if best_num_worker[pin_memory] == 0:
                    the_range = []
                else:
                    the_range = list(range(best_num_worker[pin_memory] - 3, best_num_worker[pin_memory]))
                for num_workers in (
                        the_range + list(range(best_num_worker[pin_memory] + 1, best_num_worker[pin_memory] + 4))):
                    current_time = loading_time(img_loader, num_workers, pin_memory,
                                                batch_size)  #loading_time(num_workers, pin_memory)
                    if current_time < best_time[pin_memory]:
                        best_time[pin_memory] = current_time
                        best_num_worker[pin_memory] = num_workers
                break
    if best_time[0] < best_time[1]:
        print("Best num_workers =", best_num_worker[0], "with pin_memory = False")
    else:
        print("Best num_workers =", best_num_worker[1], "with pin_memory = True")
    return
