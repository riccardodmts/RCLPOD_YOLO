import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import matplotlib as mpl
from copy import deepcopy
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams.update({'font.size': 10}) 

colors = ["#5DADE2", "#FF5433", "#F1C40F", "#1E8449", "#F39C12", "#5F6A6A", "#7D3C98", "#0F34B8", "#922B21", "#2C3E50"]


def read_csvs(results_dir = "./results", num_tasks = 1, mAP50=False):

    to_add = "50" if mAP50 else ""
    dataframes = [pd.read_csv(results_dir + f"/mAPs{to_add}_task_{task}.csv", sep="\t") for task in range(num_tasks)]
    return dataframes


def aggregate_last_results(tasks_results):

    """Get np array with mAPs of all tasks [num_tasks, num_cols]"""

    aggregated_results = np.zeros((len(tasks_results), len(tasks_results[0].columns)), dtype=np.float32)

    for idx, task_results in enumerate(tasks_results):

        aggregated_results[idx, :] = task_results.iloc[-1].to_numpy()

    return aggregated_results



def compute_mAP_groupdby(results, groups):
    """Get mAP at each task for each group of classes: [num_tasks, num_groups]"""

    mAPs = np.zeros((len(results), len(groups)), dtype=np.float32)

    for idx in len(results):

        APs = results[idx,:]

        for j, group in enumerate(groups):
            mAPs[idx, j] = np.sum(APs[group])/len(group)

    return mAPs

def compute_mAP_per_task(results, groups):

    mAPs = np.zeros(len(results), dtype=np.float32)

    for i, (APs, group) in enumerate(zip(results, groups)):
        mAPs[i] = np.sum(APs[group])/len(group)

    return mAPs

def compute_mAP_per_task_cum(results, groups):
    num_mAPs = sum([len(g) for g in groups[1:]]) + 1 if isinstance(groups[1][0], list) else len(results)
    mAPs = np.zeros(num_mAPs, dtype=np.float32)
    cum_idx = 0
    for i, (APs, group) in enumerate(zip(results, groups)):
        if isinstance(groups[1][0], list) and i>0:
            for j,g in enumerate(group):
                mAPs[cum_idx + j] = np.sum(APs[g])/len(g)
            cum_idx += len(group)
        else:
            mAPs[i] = np.sum(APs[group])/len(group)
            cum_idx+=1

    return mAPs


def mAP_exps(exp_folders, groups, limit_tasks, mAP50=False):

    exp_results = [aggregate_last_results(read_csvs(exp, limit_tasks, mAP50)) for exp in exp_folders]

    exp_mAPs = []

    for exp in exp_results:

        exp_mAPs.append(compute_mAP_per_task(exp, groups))

    return exp_mAPs

def mAP_exps_cum(exp_folders, groups, limit_tasks, mAP50=False):

    exp_results = [aggregate_last_results(read_csvs(exp, limit_tasks, mAP50)) for exp in exp_folders]

    exp_mAPs = []

    for exp in exp_results:

        exp_mAPs.append(compute_mAP_per_task_cum(exp, groups))

    return exp_mAPs

def generate_groups(type_exp="15p1", is_coco=False):

    total_classes = 80 if is_coco else 20
    columns_per_task = []

    num_classes_first_task, jump = type_exp.split("p")
    jump = int(jump)
    num_classes_first_task = int(num_classes_first_task)

    num_tasks = (total_classes - num_classes_first_task) // jump + 1
    init_classes = 2

    for i in range(num_tasks):

        stop = int(num_classes_first_task)+2 if i == 0 else (init_classes + int(jump))

        if i>0:
            columns_per_task.append(columns_per_task[-1]+list(range(init_classes, stop)))
        else:
            columns_per_task.append(list(range(init_classes, stop)))

        init_classes = stop

    return columns_per_task



def plot_mAP_exps(type_exp, is_coco, exp_folders, labels, mAP50=False, limit_tasks=None, save=None, baseline=None):

    groups = generate_groups(type_exp, is_coco)
    num_tasks = len(groups) if limit_tasks is None else limit_tasks

    mAPs = mAP_exps(exp_folders, groups[:num_tasks], num_tasks, mAP50)

    fig, ax = plt.subplots()
    for i in range(len(labels)):
        ax.plot(mAPs[i] * 100, "o-", label = labels[i], color=colors[i])

    if baseline:
        baseline_groups = []
        for g in groups:
            for i in g:
                if i not in baseline_groups:
                    baseline_groups.append(i)
    mAPs_baseline = None if baseline is None else mAP_exps([baseline], [baseline_groups], 1, mAP50)
    if baseline is not None:
        mAPs_baseline = np.repeat(mAPs_baseline, num_tasks)
        ax.plot(mAPs_baseline * 100, "-", label = "joint training", color=colors[-1])  

    ax.set_xlabel("Task")
    if not mAP50:
        ax.set_ylabel("$\\text{mAP}_{\\text{Val}}^{50-95}$")
    else:
        ax.set_ylabel("$\\text{mAP}_{\\text{Val}}^{50}$")

    xticks = np.arange(0, num_tasks, 1)
    xlabels = [f'{x+1}' for x in xticks]
    ax.set_xticks(xticks, labels=xlabels)
    plt.legend()
    if save is None:
         plt.show()
    else:
        plt.savefig(save, bbox_inches='tight')

def plot_mAP_exps_task(type_exp, is_coco, exp_folders, labels, task_id, mAP50=False, limit_tasks=None, save=None, baseline=None):

    groups = generate_groups(type_exp, is_coco)
    group = groups[task_id]


    num_tasks = len(groups) if limit_tasks is None else limit_tasks
    temp_group = groups[task_id]
    group = []

    if task_id > 0:
        for idx in temp_group:
            if idx in groups[task_id - 1]:
                pass
            else:
                group.append(idx)
    else:
        group = temp_group
    group = [group]

    groups = group * num_tasks

    mAPs = mAP_exps(exp_folders, groups[:num_tasks], num_tasks, mAP50)
    mAPs_baseline = None if baseline is None else mAP_exps([baseline], groups, 1, mAP50)

    fig, ax = plt.subplots()
    for i in range(len(labels)):
        ax.plot(mAPs[i] * 100, "o-", label = labels[i], color=colors[i])
    if baseline is not None:
        mAPs_baseline = np.repeat(mAPs_baseline, num_tasks)
        ax.plot(mAPs_baseline * 100, "-", label = "joint training", color=colors[-1])   

    ax.set_xlabel("Task")
    if not mAP50:
        ax.set_ylabel("$\\text{mAP}_{\\text{Val}}^{50-95}$")
    else:
        ax.set_ylabel("$\\text{mAP}_{\\text{Val}}^{50}$")

    xticks = np.arange(0, num_tasks, 1)
    xlabels = [f'{x+1}' for x in xticks]
    ax.set_xticks(xticks, labels=xlabels)
    plt.legend()
    if save is None:
        plt.show()
    else:
        plt.savefig(save, bbox_inches='tight')

    return num_tasks



def build_table(type_exp, is_coco, exp_folders, mAP50=False):

    """return: numpy array [N experiments x M], M= 1 + 3 * (num_tasks-1)
    3 because: one for the mAP old classes, one for the overall mAP and one for just the new classes"""

    groups = generate_groups(type_exp, is_coco)
    num_tasks = len(groups)

    tasks_groups = []

    for task_id in range(num_tasks):
        group = []
        temp_group = groups[task_id]

        if task_id > 0:
            for idx in temp_group:
                if idx in groups[task_id - 1]:
                    pass
                else:
                    group.append(idx)
        else:
            group = temp_group
        if task_id>0:
            old_group = tasks_groups[-1][1] if task_id > 1 else tasks_groups[-1]
            val_group = old_group + group
            tasks_groups.append([old_group, val_group, group])
        else: 
            tasks_groups.append(group)

    mAPs = mAP_exps_cum(exp_folders, tasks_groups, num_tasks, mAP50)
    return np.stack(mAPs), num_tasks


def table_to_csv(type_exp, is_coco, exp_folders, labels, mAP50=False):

    array, num_tasks = build_table(type_exp, is_coco, exp_folders, mAP50)
    head = ["Task 1"]
    for i in range(num_tasks-1):
        head.append(f"Task {i+2} - old")
        head.append(f"Task {i+2} - all")
        head.append(f"Task {i+2} - new")

    dataframe = pd.DataFrame(
        data = array,
        columns=head,
        index = labels
    )
    return dataframe








    











