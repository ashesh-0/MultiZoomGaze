"""
Functions helping analysis of lstm vs static model.
"""
import matplotlib.pyplot as plt


# session:56, person:291 has very large error for LSTM
# Session:75 Person:37
def get_filtered_dfs(start_frame, end_frame, session, person, model1_df, model2_df):
    # print(f'Session:{session} Person:{person}')
    model1 = model1_df[(model1_df.person == person) & (model1_df.session == session)]
    model2 = model2_df[(model2_df.person == person) & (model2_df.session == session)]
    assert model2.frame.equals(model1.frame)
    if start_frame:
        model2 = model2[model2.frame >= start_frame]
        model1 = model1[model1.frame >= start_frame]

    if end_frame:
        model2 = model2[model2.frame <= end_frame]
        model1 = model1[model1.frame <= end_frame]

    return model1, model2


def plot_model1_model2(model1,
                       model2,
                       model1_col='angular_err_avg',
                       model2_col='angular_err_avg',
                       yaw=False,
                       pitch=False,
                       ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(20, 5))
    if yaw or pitch:
        ax1 = ax.twinx()
        if yaw:
            model1.set_index('frame')['g_yaw'].plot(ax=ax1, c='r', label='Yaw', linestyle=':', alpha=0.5)
        if pitch:
            model1.set_index('frame')['g_pitch'].plot(ax=ax1, c='blue', label='Pitch', linestyle=':', alpha=0.5)
        ax1.legend(loc='upper right')

    model1.set_index('frame')[model1_col].plot(ax=ax, label='model1', marker='.')
    model2.plot(x='frame', y=model2_col, ax=ax, label='model2', marker='.')
    _ = ax.legend(loc='upper left')
    return ax
