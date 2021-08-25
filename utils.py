import numpy as np
from torch import as_tensor, float
from proposed_design_types import PROPOSED_DESIGN_TYPES
from sklearn.model_selection import train_test_split


def calculate_pos_weights(class_counts,data):
    pos_weights = np.ones_like(class_counts)
    neg_counts = [len(data)-pos_count for pos_count in class_counts]
    for cdx, (pos_count, neg_count) in enumerate(zip(class_counts,  neg_counts)):
        pos_weights[cdx] = neg_count / (pos_count + 1e-5)

    return as_tensor(pos_weights, dtype=float)


def add_ptyp_vecs_calc_pos_weights(data):
    ptyps = data['ptyp']

    y = np.zeros((len(data), len(PROPOSED_DESIGN_TYPES)), dtype=int)
    class_counts = np.zeros((len(PROPOSED_DESIGN_TYPES), 1), dtype=np.int)

    for i in range(len(data)):
        for j in range(len(PROPOSED_DESIGN_TYPES)):
            if PROPOSED_DESIGN_TYPES[j] in ptyps[i]:
                y[i, j] = 1
                class_counts[j] += 1
    pos_weight = calculate_pos_weights(class_counts, data)

    return pos_weight, y


def get_val_train_data(data, y, data_type, val_size, seed):
    # ptyp_vecs = data['ptyp_vec']
    ptyp_vecs = y
    titles = data['title']
    abstracts = [eval(abs)[0]['text'] if eval(abs) else '' for abs in data['abstract']]

    if data_type == "abstracts":
        x_train, x_val, y_train, y_val = train_test_split(abstracts, ptyp_vecs, test_size=val_size, random_state=seed)
    elif data_type == "titles":
        x_train, x_val, y_train, y_val = train_test_split(titles, ptyp_vecs, test_size=val_size, random_state=seed)
    elif data_type == "titles_abstracts":
        combined = ["{} {}".format(t, a) for t, a in zip(titles, abstracts)]
        x_train, x_val, y_train, y_val = train_test_split(combined, ptyp_vecs, test_size=val_size, random_state=seed)

    return (x_train, y_train), (x_val, y_val)