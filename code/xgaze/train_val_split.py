from xgaze.kfold_split import KFOLD_SPLIT


def get_train_val_split(kfold_id=None):
    """
    3 specs, 7 girls in validation set.
    """
    if kfold_id is not None:
        return get_Kfold_train_val_split(kfold_id)

    val = [
        'subject0006.h5', 'subject0007.h5', 'subject0008.h5', 'subject0024.h5', 'subject0027.h5', 'subject0030.h5',
        'subject0040.h5', 'subject0046.h5', 'subject0056.h5', 'subject0078.h5', 'subject0084.h5', 'subject0085.h5',
        'subject0101.h5', 'subject0103.h5', 'subject0109.h5'
    ]

    train = [
        'subject0000.h5', 'subject0003.h5', 'subject0004.h5', 'subject0005.h5', 'subject0009.h5', 'subject0010.h5',
        'subject0013.h5', 'subject0014.h5', 'subject0015.h5', 'subject0016.h5', 'subject0018.h5', 'subject0019.h5',
        'subject0021.h5', 'subject0026.h5', 'subject0028.h5', 'subject0029.h5', 'subject0031.h5', 'subject0032.h5',
        'subject0033.h5', 'subject0035.h5', 'subject0036.h5', 'subject0038.h5', 'subject0039.h5', 'subject0041.h5',
        'subject0043.h5', 'subject0044.h5', 'subject0045.h5', 'subject0048.h5', 'subject0050.h5', 'subject0051.h5',
        'subject0052.h5', 'subject0055.h5', 'subject0057.h5', 'subject0058.h5', 'subject0059.h5', 'subject0060.h5',
        'subject0061.h5', 'subject0062.h5', 'subject0063.h5', 'subject0065.h5', 'subject0066.h5', 'subject0067.h5',
        'subject0069.h5', 'subject0072.h5', 'subject0073.h5', 'subject0075.h5', 'subject0076.h5', 'subject0079.h5',
        'subject0080.h5', 'subject0081.h5', 'subject0083.h5', 'subject0088.h5', 'subject0090.h5', 'subject0092.h5',
        'subject0095.h5', 'subject0098.h5', 'subject0099.h5', 'subject0100.h5', 'subject0102.h5', 'subject0104.h5',
        'subject0105.h5', 'subject0106.h5', 'subject0107.h5', 'subject0108.h5', 'subject0111.h5'
    ]
    return train, val


def get_Kfold_train_val_split(kfold_id):
    print('Using Kfold training, kfold_id=', kfold_id)
    assert kfold_id in [0, 1, 2, 3, 4]
    val = KFOLD_SPLIT[kfold_id]
    train = []
    for i in range(5):
        if i == kfold_id:
            continue
        train += KFOLD_SPLIT[i]
    return train, val
