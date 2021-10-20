# Objective
In this experiment, we model yaw angle as
1. sin(yaw) along with backward/forward boolean signal.
2. cos(yaw) along with left/right boolean model.
3. sin(yaw) along with cos(yaw). Here, we predict yaw in two ways. In the first, we use the predicted sin(yaw) and the
    sign of predicted cos(yaw) to estimate yaw. In the second, we use the predicted cos(yaw) and the sign of predicted
    sin(yaw) to predict the yaw.
4. sin(yaw) along with cos(yaw). Estimation of yaw is done as in previous model. We regularize the prediction by adding
 an MSE loss on sin2(yaw) + cos2(yaw) = 1

In all cases, the discontinuity problem is not there. Although, discontinuity exists in boolean signal. However, the incorrect prediction there does not lead to that different prediction. It is because if we are at an angle theta yaw from the boundary, then the error is 2*theta. However, in case of original formulation, it could be anything.


# Inference:
We see that using sin(yaw),cos(yaw) along with regularization gives best performance for static model. We train LSTM
on this setting and get 4.8% improvement in gaze prediction.

# Issues:
We see that the model is having difficulty predicting 1/-1. When we use cos(yaw) to get yaw, yaw=0 is underpredicted.
When we use sin(yaw) to get yaw, yaw=-pi/2,pi/2 is underpredicted.

## Using this knowledge about difficulty in predicting 1.
In model 3 and 4, we use this knowledge to better combine the two predictions of yaw. We take final prediction of yaw
to be the weighted average of the two yaw predictions. The weight for yaw estimated from predicted sin(yaw)
is cos(avg yaw prediction). When actual yaw is 0, then cos model will perform badly and so we ignore it altogether.
When actual yaw is pi/2,-pi/2, sin model will perform badly and so we ignore it altogether.


## Experiment:
I tried to see if reducing fc1, fc2 does anything to the model. From 1000 => 256,
one can reduce it to 256=> 128 without much if any loss of accuracy.
1000 => 128 gave exactly same accuracy as 1000=>256

Config                      Best Validation         Train angular error on 99 epoch
fc1:1000,fc2:16             14.04                   1.57
fc1:1000,fc2:32             14.02                   1.51
fc1:1000,fc2:64             14.02                   1.51
fc1:1000,fc2:128            13.95                   1.49
fc1:500,fc2:128             14.07                   1.50
fc1:250,fc2:128             14.11                   1.53
fc1:128,fc2:128             14.06                   1.52
