# Gaze360: Physically Unconstrained Gaze Estimation in the Wild Dataset

## About

This is code for training and running our Gaze360 model. The usage of this code is for non-commercial research use only. By using this code you agree to terms of the [LICENSE](https://github.com/Erkil1452/gaze360/blob/master/LICENSE.md). If you use our dataset or code cite our [paper](x) as:

 > Petr Kellnhofer*, Adrià Recasens*, Simon Stent, Wojciech Matusik, and Antonio Torralba. “Gaze360: Physically Unconstrained Gaze Estimation in the Wild”. IEEE International Conference on Computer Vision (ICCV), 2019.

```
@inproceedings{gaze360_2019,
    author = {Petr Kellnhofer and Adria Recasens and Simon Stent and Wojciech Matusik and and Antonio Torralba},
    title = {Gaze360: Physically Unconstrained Gaze Estimation in the Wild},
    booktitle = {IEEE International Conference on Computer Vision (ICCV)},
    month = {October},
    year = {2019}
}
```

## Data
You can obtain the Gaze360 dataset and more information at [http://gaze360.csail.mit.edu](http://gaze360.csail.mit.edu).

This repository provides already processed txt files with the split for training the Gaze360 model. The txt contains the following information:
* Row 1: Image path
* Row 2-4: Gaze vector

Note that these splits only contain the samples which have available a one second window in the dataset.

## Requriments
The implementation has been tested wihth PyTorch 1.1.0 but it is likely to work on previous version of PyTorch as well.


## Structure

The code consists of
- This readme.
- The training/val/test splits to train the Gaze360 model, as described in the Data section.
- The model and loss definition (model.py)
- A script for training and evaluation of the Gaze360 model (run.py).
- A data loader specific for the Gaze360 dataset (data_loader.py)

## Trained models

The model weights can be downloaded from this [link](http://gaze360.csail.mit.edu/files/gaze360_model.pth.tar)

## Gaze360 in videos
A beta version of the notebook describing how to run Gaze360 on Youtube videos is now [online](https://colab.research.google.com/drive/1AUvmhpHklM9BNt0Mn5DjSo3JRuqKkU4y)!


## Ideas
1. Freeze the base network of lstm. apply data augmentations whereby
    a. one reverses the time.
    b. go back an forth in time.
    c. increase sequence length.
2. Try to find difference in model performance between lstm and normal model. video plots of errors. worst images.
50% quantile images.
3. DONE.     I don't see reasonable numbers for static best model. I think I had got 13.95 somewhere.



## Difference in performance btw lstm and static model.
1. Predictions are smoother in LSTM. **IDEA** is to smoothen the static model predictions. It works
2. See sections of video where LSTM is doing better, where static is doing better.
3. Question: Where are the missing frames.: There are no missing frames. Some frames are just not used for
    training/validation. However, for LSTM, all 7 frames are consequitive.
4. Plot camera-person distance.
5. Look at 70-80, 80-90 percentile videos and plots to see if one can say why the performance is bad for some.
6. Magnification factor:
    a. uniformly increase the crop to see the effect.<br/>
    b. randomly pick few frames to magnify.<br/>
    c. ~~remove jerks in frames and then predict.~~<br/>

### Normal Magnification: LSTM:
Procedure: Resize(224) => CenterCrop(centercropsize) => Resize(224) => img_normalize
CenterCropsize  ValidationError<br/>
Nocrop     12.25<br/>
210        12.12<br/>
200        12.20<br/>
175        12.75<br/>
150        14.28<br/>

### RandomCenterCrop Magnification: LSTM:
Procedure: with probablity 0.5 do the Normal Magnification for each frame.
Cropsize  ValidationError<br/>
Nocrop     12.25<br/>
210        12.15<br/>
200        12.16<br/>
175        12.21<br/>
150        12.49<br/>

### OneInSequenceMagnification: LSTM:
Procedure: Perform Normal Magnification of kth image in the sequence of 7 images.

     K=1    K=2    K=3    K=4    K=5    K=6   K=7
125 12.25  12.21  12.17  12.15<br/>
150 12.23  12.18  12.14  12.11  12.13  12.16 12.19<br/>
175 12.24  12.20  12.16  12.12 <br/>
200 12.24  12.22  12.19  12.17

### Normal Magnification on static model:
Cropsize    ValidationError
150         tensor(17.2649)
175         tensor(15.0405)
200         tensor(14.0755)
205         tensor(14.0215)
210         tensor(13.9473)
215         tensor(13.9198)
220         tensor(13.9135)
224         tensor(13.8879)

### Theoretically best performance from multiple Normal Magnifications on static model:
Here we try multiple crop sizes. For each instance, we pick the best crop size. So it gives a theoretical best
performance assuming we can predict which cropsize to pick for each image.

Using [100, 125, 150, 175, 200, 210, 225] crops, the result is 9.66
Using [125, 150, 175, 200, 210, 225] crops, the result is 10.0
Using [150, 175, 200, 210, 225] crops, the result is 10.47
Using [175, 200, 210, 225] crops, the result is 11.15
Using [200, 210, 225] crops, the result is 12.04
Using [210, 225] crops, the result is 12.7
Using [225] crops, the result is 13.96


### Ideas on leveraging the Magnifications.
0. Train with centercrop
1. convert image to gray scale, 3 dimensions comprises of different scales. may be increase 3 to a higher number.
2. data augmentation: randomly zoom in a lot or zoom out a bit. in zooming out, one needs to fill it with zeros/noise.
3. data augmentation specifically for lstm: 7 frames have different zoom levels. (This improves things)
4. We know that blurring an image affects the initial layer weights. so one can train with multiple input branches.each branch can take different zoom level input. This stablizes the final layers. One can then train with one input branch freezing the final layers. Attention mechanism may be used here to ensure that the best zoom branch is taken into consideration. Attention may not work as low level features may not contain the information needed for selection.
5. search the literature for what has been done on magnification.
6. take output from multiple layers as features. However, problem here is that we are talking about 224 going to 150. One maxpool operation does atleast 2x. However, our scale is < 2.
7. for normally trained model, look at the heatmap to classify the best zoom level. Doesn't work. it gives trivial majority prediction.
8. Spatial Transformer: Look at distribution of predicted transformations
9. Spatial Transformer: increase the learning rate
10. Spatial Transformer: apply severe Centercrop augmentation  so as to force the model to learn non trivial STN

#### Magnification Analysis of LSTM model with differential cropping
1. Evaluate without any cropping. Since LSTM doesn't have time specific weights, LSTM then has learnt to handle features from all level of crops. So without cropping, the performance should not be that bad.<br/>
            a. 12.06 is with [224,200,175,150,175,200,224]. This is used in training.<br/>
            b. 14.53 is with no cropping, or [224]*7<br/>
            c. 12.6139 with [200]*7 cropping<br/>
            d. 12.36 with [175]*7 cropping<br/>
            e. 13.21 with [150]*7 cropping<br/>
            f. 12.12 with [200, 200, 175, 150, 175, 200, 200] cropping<br/>
            g. 12.20 with [224, 224, 175, 150, 175, 224, 224] cropping<br/>
            i. 12.10 with [224, 175, 175, 150, 175, 175, 224] cropping<br/>
            j. 12.11 with [224, 200, 150, 150, 150, 200, 224] cropping<br/>
            k. 12.29 with [224, 200, 200, 150, 200, 200, 224] cropping<br/>
            l. 12.13 with [224, 200, 175, 175, 175, 200, 224] cropping<br/>
            <!-- Reordered cropping -->
            m. 12.66 with [175, 200, 224, 150, 224, 200, 175] cropping<br/>
            n. 12.31 with [224, 200, 175, 150, 224, 200, 175] cropping<br/>
    **Inferences**: The head of the model and the 2nd layer of LSTM has been trained to expect features from all levels.
    That is the reason, we see inferior performance. When we compare (d) 175 single crop with m. reverse order of crops,
    we see the significance of the order. even though all levels of crop is present in m. just due to ordering, a lot of performance degrades. Same can be said when looking at m. and n.


#### Things to try related to Magnification May26.
1. Try batch size of 64 on differential cropping. The best LSTM model created in march had batch_size 64. It indeed improves the performance.
2. Try reversing the time.

Facts:
1. Validation performance improves a bit when images are zoomed in.
This means that there are images in validation which are too small. That is model is trained to evaluate features of slightly zoomed in images.

How to fix it:
1. Make Validation images zoom in. It gave marginal improvement.
2. Use differential zooming to pass in images of multiple zoomed images.
2. Make Training more robust by traing with zoomed out images as well.
3. Given an image, find where to focus so as to have appropriate zoom level.

I'm thinking about 3. Using spatial transformer.
Current analysis of Transformer:
1. With zoom in and out data transformation, transformer is learning something. however, I don't see any evidence of stn causing better performance. When I inspect magnification < 0.95, no stn is better. So high magnification numbers on validation set is not resulting in better performance. Separately, when I look at distribution of magnification in cases where stn is better as opposed to cases where no stn is better, I get identical distribution on magnification. This is telling that the model is overfitting to an extent that transformer is not able to improve results for extreme magnifications. Note that overall, using stn helps.  Also, given the fact that min magnification is 0.88 tells us that magnification can certainly be more low.

=> Train Spatial Transformer so as to zoom in and focus on image at hand. If it is done decently, then one can fit it
in the static model/lstm model with no data augmentation.
=> Try to look for which images we see magnification is significant. and for which it isn't.

#### Other avenues:
1. We can look for image rotation based. We can give slight rotations about eye center. And crop slightly so as to ensure edges are not seen. As for target gaze, z component will not change. x,y will change with same angle.
2. It does not make sense why reversing the time does give poor results. 12.15 is the number.

#### Things to try related to Magnification May30.
1. Pre-train spatial transformer. Since training is overfitting, that is avoiding the network to learn the correct zoom and shift parameters. One can pre-train it first.

#### Analysis of differential cropping June1.
1. ~~Compare LSTM with sinecosine target and LSTM with sinecosine target and differential cropping. For which range of yaw, pitch, which is better.~~ I see that differential cropping helps for back images and for extreme gazes.
2. ~~Look at individual examples where each model has excelled.~~
3. ~~Do the same thing among more extreme cropsize list.~~ Did not persue this.
4. Using extended head, run a static model with zoomed out cropping.

#### Future directions:
1. Show the generalizability of differential cropping.
2. Incorporate STN into LSTM network. One idea is to convert cropping to a layer. DiffCrop. input is passed to STN.
Output of STN is then passed through DiffCrop. Idea is that done this way, the performance of model is more sensitive to
 tx and ty.
3. Incorporate attention. get features from multiple zooms. Then use attention to combine them. This should have a
 regularizable effect as well.
4. Incorporate attention2: Create multiple input branches. combine them at some stage. This is generalization of Idea 3.
5. Try to reduce the size of STN. See if that helps or not.
6. For the static model, I see that unfreezing works. centercropping ofcourse works. we can combine both.
7. For static model, try out multiple cropping and output the median value as score.
8. search the literature for what has been done on magnification.
9. Spatial Transformer: increase the learning rate
10. One can train the model for 10 epochs using 5 fold CV. In each fold, one can then decide upon the best zoom factor and best x,y offset factor. One needs to ensure that multiple runs of this result in very similar target. One can then train the model to predict that as well. This may help to predict the x,y and relevant zoom factor needed.
11. Attention can work. Same Base model. multiple zoomed inputs. attention to get final features. dense layers.
12. Separate input branches on top of 11. That should give more benefit.
13. Attention can even be pixel wise. when the feature map is 56*56. At that point, we can may be use attention. Spatial attention.
14. Separately from all this, add gaussian noise to the crop size.
15. There are two ways of centercropping: make the boundary black. so only center has information. or expand the center to fill the whole image. Expanding it to center is what matters here. No extra parameters for static. feed it multiple scales. use attention to weight them and then use it as input to final dense layer.
16. Look at the median of the gradient of each layer. How it changes with time. That should give some idea about which layers weights change a lot. Idea is to start freezing later layers so as to overcome overfitting.

#### Reassesment June8
1. With STN, I see a lot of fluctuation in results. (both train and validation numbers)
2. Pre loading STN does not seem to matter.
3. Using Extended head (larger head image from full body image) for zoom out images does not look to improve the perf.
4. Increasing the granularity on scale does not look to improve the perf. Here, normally, only few scales are allowed.
However, when we try to randomly select scale randomly from a range, I see perf. degrade slightly.
5. With STN, however, increasing the granularity on scale does look to improve the perf.
6. With STN, smaller learning rate looks better. Best combination is STN+lower lr. If we want extended head, then  ExtendedHead+increased granularity.
Raw data:
        TYPE:9_fc1:None_fc2:128_bsz:128_lr:0.0001_stn:1_stn_pre_load:0_ccrop_list:250-198.pth.tar
        Angular (14.0820) 0.005 1.576

        TYPE:9_fc1:None_fc2:128_bsz:128_lr:0.0001_stn:0_stn_pre_load:0_ccrop_list:250-198.pth.tar
        Angular (14.1150) 0.005 1.51

        TYPE:9_fc1:None_fc2:128_bsz:128_lr:0.0001_stn:1_stn_pre_load:0_ccrop_list:250-198_rn:2.pth.tar
        100:
            Angular (13.85)    0.006    1.752
        200:
            Angular (13.7528)    0.004    1.23

        TYPE:9_fc1:None_fc2:128_bsz:128_lr:0.0001_stn:1_stn_pre_load:1_ccrop_list:250-198_rn:2.pth.tar
        100:
            Angular (14.0184)    0.005    1.67

        200:
            Angular (13.9482)    0.003    1.21

        TYPE:9_fc1:None_fc2:128_bsz:128_lr:0.0001_ccrop_list:250-198_ex_head:1.pth.tar
        100:
            Angular (14.2241)    0.007    2.05

        200:
            Angular (14.13)    0.005    1.45


        TYPE:9_fc1:None_fc2:128_bsz:128_lr:0.0001_ccrop_list:250-198_ex_head:1_fg_rand:1.pth.tar
        100:
            Angular (14.2647)    0.007    2.02
        166:
            Angular (14.2037)    0.005    1.609

        TYPE:9_fc1:None_fc2:128_bsz:128_lr:0.0001_stn:1_ccrop_list:250-198_ex_head:1_fg_rand:1.pth.tar
        200:
            Angular (13.9349)    0.0048    1.47
        166:
            Angular (13.9349)    0.005    1.6
        100:
            Angular (14.1005)    0.007    2.03

        TYPE:9_fc1:None_fc2:128_bsz:128_lr:0.0001_stn:1_ccrop_list:250-198_ex_head:1_fg_rand:0.pth.tar
        200:
            Angular (14.1171)    0.004    1.44
        100:
            Angular (14.1171)    0.007    2.04

        TYPE:9_fc1:None_fc2:128_bsz:128_lr:0.0001_stn:1_stn_lr:8_ccrop_list:250-198_ex_head:0_fg_rand:0.pth.tar
        83:
            Angular (14.4060)    0.008    2.3

        TYPE:9_fc1:None_fc2:128_bsz:128_lr:0.0001_stn:1_stn_lr:16_ccrop_list:250-198_ex_head:0_fg_rand:0.pth.tar
        83:
            Angular (14.1460)    0.008    2.31
        183:
            Angular (14.0572)    0.005    1.55

        TYPE:9_fc1:None_fc2:128_bsz:128_lr:0.0001_stn:1_stn_lr:0.5_ccrop_list:250-198_ex_head:0_fg_rand:0.pth.tar
        83:
            Angular (14.1036)    0.006    1.92
        199:
            Angular (13.8775)    0.004    1.217

        TYPE:9_fc1:None_fc2:128_bsz:128_lr:0.0001_stn:1_stn_lr:2_ccrop_list:250-198_ex_head:0_fg_rand:0.pth.tar
        82:
            Angular (14.3880)    0.008    2.32

7. After epoch 20, I get consistent values for LSTM. STN+ post crop doesn't work well.
8.

1. 16bit fc2:256 num_inputs:1 [224] 14.40
2. 16bit fc2:256 num_inputs:2 [224, 150] 13.99
3. 16bit fc2:256 num_inputs:2 mode:attention Attn_dim:120 [224, 150] 14.14
4. 16bit fc2:256 num_inputs:1 mode:cat Attn_dim:120 [224] 14.09
6. fc2:256 num_inputs:2 mode:attention Attn_dim:256 [224, 150] 13.98




### Things to do June 10
1. ~~We have a better static model using STN. Do an analysis to figure out where does it perform better. And does STN help~~ Performance on test set is poor.
2. ~~Run LSTM with multi crops as input. show that this does a better job of feature integration.~~
3. ~~Try to find out whether there is a plateau around the minimum error for zoom. Using the uncertainity to determine the zoom level looks to have some truth to it.~~Nothing significant
4. ~~Try to find out whether there is a plateau around the minimum error for zoom + tx,ty.~~ Nothing significant.

### LSTM used for scale invariance.
1. Static model performs a lot better with LSTM + centercropping.
2. Using above face, we can have 2 LSTM modules. One for temporal and one for centercropping. This is getting slow. 1 epoch in 30 min

### Things to do June 16
1. STATIC: In LSTM module, make output index configurable. Currently, it is set to middle element.
2. STATIC: Explore more finegrained cropping layers. It is because I had seen going below 150 doesn't work well. So we can get more granular till 150.
3. STATIC: Explore zoom out as well.
4. SEQUENTIAL: Take input 2 set of images. one set comprise of 6 images. another set comprises of K images for the
target frame. This way, I can extract maximum info from central frame and still use other frames.
5. Find another dataset where the LSTM module works.
6. Train on sin(theta + 45), cos(theta + 45) as well.



### Things to do in June 19
1. I'm facing difficulty in reproducing the scale invariance using LSTM + centercropping. It may be due to 3 reasons
    a) ~~Code has changed.~~ I've checked it carefully. I don't see that happening.
    b) ~~Package changes~~ I think this is mostly the issue. I've got better numbers with 0.5 torchvision, 1.5 torch
    c) Random seed. I think this may as well be the reason.
log_model10_35.txt was run with var being in [-1,1]
log_model10_33.txt and log_model10_36.txt was run with var being in [-pi,pi]

2. For showing scale invariance, we may want to skip the resize operation and see what performance do we get.

### Things to do in June 22
1. Work with epsilon pinball loss.
2. Work with pinball loss + angular loss. Angular loss allows a conical region around: Angular loss has an issue.
    One needs to ensure that sin(theta) and cos(theta) satisfy 1 constraint. Actually both of them are getting predicted
    near 1. explicitly adding the constraint leads to another bad training error.

Best:
            log_model10_23.txt
            gaze360_static_TYPE:10_fc2:256_time:False_diff_crop:224-150_bsz:64_lr:0.0001.pth.tar
            [ImageLoader] reverse:False random_walk:False forward_bias:0.7 Time:False
            [ImageLoaderSineCosineMultiSizedCrops] Cropsizes:[224, 200, 175, 150, 175, 200, 224]
            Train: 1.353,0.0045
            Angular (13.5333) Loss (0.0876)
            Angular (13.5348) Loss (0.0884)
            Angular (13.5502) Loss (0.0887)
            Epoch67:
                Angular (13.5866) Loss (0.0869)
                Angular (13.5906) Loss (0.0839)
                Angular (13.6256) Loss (0.0870)
            Epoch52:
                Train:1.868,0.0064
                Angular (13.5906) Loss (0.0839)
                Angular (13.6264) Loss (0.0843)
                Angular (13.6447) Loss (0.0846)
            Epoch30:
                Angular (13.7860) Loss (0.0824)
                Angular (13.8144) Loss (0.0819)
                Angular (13.8557) Loss (0.0786)
            Epoch10:
                Angular (14.1581) Loss (0.0681)
                Angular (14.4267) Loss (0.0727)
                Angular (14.5560) Loss (0.0689)


            log_model10_27.txt
            gaze360_static_TYPE:10_fc2:256_time:False_diff_crop:224-150_tar_idx:3_seq_len:7_bsz:64_lr:0.0001.pth.tar
            [ImageLoader] reverse:False random_walk:False forward_bias:0.7 Time:False SeqLen:7
            [ImageLoaderSineCosineMultiSizedCrops] Cropsizes:[224, 200, 175, 150, 175, 200, 224]
            Epoch52:
                Train:2.1,0.0064
                Angular (13.7875) Loss (0.0855)
                Angular (13.7997) Loss (0.0864)
                Angular (13.8203) Loss (0.0843)
            Epoch30:
                Angular (13.9540) Loss (0.0849)
                Angular (13.9665) Loss (0.0838)
                Angular (13.9865) Loss (0.0802)

            Epoch10:
                Angular (14.3876) Loss (0.0747)
                Angular (14.4149) Loss (0.0694)
                Angular (14.4777) Loss (0.0670)

            log_model10_33.txt
            gaze360_static_TYPE:10_fc2:256_time:False_diff_crop:224-150_bsz:64_lr:0.0001_v:7.pth.tar
            [GazeSinCosLSTM] Freeze:0 STN:False STN_clist:None
            train.txt
            [ImageLoader] reverse:False random_walk:False forward_bias:0.7 Time:False
            [ImageLoaderSineCosineMultiSizedCrops] Cropsizes:[224, 200, 175, 150, 175, 200, 224]
            Epoch64:
                1.651(0.0056)
                Angular (13.6821) Loss (0.0884)
                Angular (13.7039) Loss (0.0888)
                Angular (13.7247) Loss (0.0883)

            Epoch52:
                Train:1.823 0.0062
                Angular (13.7992) Loss (0.0879)
                Angular (13.8013) Loss (0.0871)
                Angular (13.8179) Loss (0.0875)

            Epoch30:
                Angular (13.8239) Loss (0.0828)
                Angular (14.0184) Loss (0.0820)
                Angular (14.0514) Loss (0.0835)
            Epoch10:
                Angular (14.1671) Loss (0.0682)
                Angular (14.2562) Loss (0.0719)
                Angular (14.4563) Loss (0.0707)

            log_model10_36.txt
            Epoch30:
                Angular (13.9389) Loss (0.0818)
                Angular (13.9592) Loss (0.0833)
                Angular (13.9995) Loss (0.0837)
            Epoch10:
                Angular (14.3791) Loss (0.0732)
                Angular (14.4688) Loss (0.0709)
                Angular (14.6072) Loss (0.0632)
            log_model10_38.txt bad.


            log_model10_39.txt
            Epoch10:
                Angular (14.3717) Loss (0.0741)
                Angular (14.4354) Loss (0.0662)
                Angular (14.6541) Loss (0.0684)
            Epoch30:
                Angular (13.8739) Loss (0.0797)
                Angular (13.9027) Loss (0.0822)
                Angular (13.9532) Loss (0.0799)
            Epoch52:
                Angular (13.6135) Loss (0.0864)
                Angular (13.6919) Loss (0.0859)
                Angular (13.6949) Loss (0.0870)
            Epoch89:
                Angular (13.6021) Loss (0.0904)
                Angular (13.6135) Loss (0.0864)
                Angular (13.6164) Loss (0.0912)





log_model10_24.txt
gaze360_static_TYPE:10_fc2:256_time:False_diff_crop:224-164_bsz:60_lr:0.0001.pth.tar
[ImageLoaderSineCosineMultiSizedCrops] Cropsizes:[224, 204, 184, 164, 144, 124, 100]
Train:1.393,0.0046
Angular (13.6854) Loss (0.0868)
Angular (13.7303) Loss (0.0906)
Angular (13.7444) Loss (0.0907)
Epoch 67:
    Angular (13.6854) Loss (0.0868)
    Angular (13.7581) Loss (0.0889)
    Angular (13.7712) Loss (0.0883)


log_model10_26.txt
gaze360_static_TYPE:10_fc2:256_time:False_diff_crop:224-150_tar_idx:3_seq_len:4_bsz:64_lr:0.0001.pth.tar
[GazeSinCosLSTM] Freeze:0 STN:False STN_clist:None TargetIdx:3
[ImageLoader] reverse:False random_walk:False forward_bias:0.7 Time:False SeqLen:4
[ImageLoaderSineCosineMultiSizedCrops] Cropsizes:[224, 200, 175, 150]
Train: 1.392, 0.0047
Angular (13.9500) Loss (0.0930)
Angular (13.9614) Loss (0.0934)
Angular (13.9880) Loss (0.0921)
Epoch67:
    Angular (14.0145) Loss (0.0911)
    Angular (14.0232) Loss (0.0893)
    Angular (14.0484) Loss (0.0919)

log_model10_25.txt
gaze360_static_TYPE:10_fc2:256_time:False_diff_crop:224-187_tar_idx:6_bsz:64_lr:0.0001.pth.tar
[GazeSinCosLSTM] Freeze:0 STN:False STN_clist:None TargetIdx:6
[ImageLoaderSineCosineMultiSizedCrops] Cropsizes:[224, 212, 199, 187, 174, 162, 150]
Epoch67:
    Train:1.565,0.0053
    Angular (13.7521)
    Angular (13.8216)
    Angular (13.8572)

log_model10_34.txt is timeseries. it gives same performance as original log_model10_10.txt

### Stability tests(branch is may21).
conda update -n base -c defaults conda
conda install torchvision cudatoolkit=10.0 -c pytorch
conda install -c conda-forge tensorboardx tqdm
conda install pandas
conda install -c anaconda scikit-image

A bug was there in following runs. Resize to 224 was not being done in transforms.
    1. torchvision:0.6.0, pytorch:1.5, cudatoolkit separate with CUDA_HOME set to 10.2 log_model9_55,log_model9_56,python 3.8.1
    2. torchvision:0.5.0, pytorch:1.4, cudatoolkit 10.1 installed as part of conda., python 3.8.1 log_model9_57,log_model9_58
    BETTER: 3. torchvision:0.5.0, pytorch:1.4, cudatoolkit 10.1 installed as part of conda., python 3.8.0 log_model9_59,log_model9_60
    4. torchvision:0.5.0, pytorch:1.4, cudatoolkit 10.0 installed as part of conda., python 3.8.0 log_model9_61,log_model9_62

1. Python 3.8.1 torch==1.4.0 torchvision==0.5.0 cudatoolkit 10.1,10.0 installed as part of conda. log_model9_65,66.txt
2. Python 3.8.0 torch==1.4.0 torchvision==0.5.0 cudatoolkit 10.1 installed as part of conda log_model9_64,63 (13.80,13.97)
3. cl6: **Python 3.8.1 torch==1.4.0 torchvision==0.5.0 cudatoolkit 10.0** installed as part of conda. log_model9_67,68.txt
4. Python 3.8.1 torch==1.5.1 torchvision==0.6.0a0+35d732a cudatoolkit-10.2 installed as part of conda. log_model9_69,70.txt
Model configuration:
/home/ashesh/gaze360_static_TYPE:9_fc1:None_fc2:128_bsz:64_centercrop:175_v:10.pth.tar
[StaticSinCosModel] fc1:None fc2:128
Overall Adam Optimizer

Inference:
3. Python 3.8.1 torch==1.4.0 torchvision==0.5.0 cudatoolkit 10.0 is the best one since it gives us super fast convergence
as compared to others.
        63,64               67,68               69,70
Epochs
10      14.57,14.50         14.54,14.37         14.55,14.70
20      14.39,14.06         14.12,13.89         14.15,14.25
30      14.14,14.03         13.94,13.89         14.09,14.20
40      14.14,14.01         13.94,13.86         14.05,14.15
50      14.09,14.01         13.88,13.83         13.95,14.04
60      13.99,14.01         13.82,13.83         13.85,14.04
70      13.99,13.94         13.73,13.72         13.83,13.96
80      13.97,13.83         13.72,13.66         13.81,13.88
90      13.97,13.80         13.69,13.66         13.75,13.87
100     13.97,13.80         13.68,13.63         13.75,13.87

## Inference June28
With latest master branch, with same configuration as 67,68, we have 71,72
TYPE:9_fc1:None_fc2:128_bsz:64_lr:0.0001_centercrop:175_v:master_2.pth.tar
Epoch    71,72
20      14.29,14.17
40      13.97,13.88
60      13.91,13.77
80      13.74,13.71
100     13.74,13.71
**This proves that current master is without any bug for this configuration.**

With almost same configuration as 71,72,67,68, we run 74,76. Difference is that we use a larger image_dimension. In 76, we use python 3.8.0. We can see that 76 performs badly as compared to 74.
Epoch       74(cl6),   76(cl6)
20          14.27,14.49
40          13.92,14.09
60          13.87,13.99
80          13.85,13.99
100         13.78,13.96
**This indicates that larger fc2 dimension is slighly worse and python 3.8.0 should not be used over 3.8.1 in our setting**


### Trying out LSTM based fusion.
With cropsizes: [224, 200, 175, 150, 175, 200, 224] and predicting from middle index.
log_model10_42.txt, log_model10_43.txt
Epoch       42,43
20          13.79,14.09
40          13.67,13.93
60          13.58,13.73
80          13.52,13.69
100         13.52,13.69


Cropsizes: [200, 175], fc2 as 128 log_model10_45
Cropsizes: [200, 175], fc2 as 256 log_model10_46,47
Epoch       45,         46,47
20          14.47,      14.22,14.19
40          14.10,      13.90,13.87
60          13.84,      13.80,13.78
80          13.77,      13.80,13.66
100         13.70,      13.77,13.66


Cropsizes: [200,175,200] Target index 1, fc2:256 log_model10_48,49
Epoch       48,49
20          14.18,14.11
40          13.94,13.84
60          13.75,13.84
80          13.75,13.77
100         13.74,13.77


Cropsizes: [200, 175] Unfreezing after epoch 1.
Epoch       50,51
20          14.13,14.23
40          13.86,13.99
60          13.86,,13.97
80          13.86,13.96
100         13.86,13.87
**Inference is that freeze-unfreeze doesn't work well here.**


### Paper ideas: June 30
1. Use tsne to plot features extracted from multiple scales. Or plot the difference of features. Idea is to figure out what is changing across multiple scales.
2. Find out whether there is benefit in using symmetry or not. If there is, then what is the role of bi-directional LSTM
Bi-directional LSTM does not integrate the information within. Rather, it .
3. Is it related to larger magnitude gradient in feature extractor. Or stable gradient?
4. Eye landmarks not feasible for extreme gazes.

## Effect of regularizer:
1. I see that regularization is not helping on average for baseline model.
2. However, for centercrop 175, regularization helps.


## Sensitivity to Zoom for Static models: LSTM multi crop vs no LSTM
In this approach, we have incorrectly centercropped the images in the case of LSTM models. For example, if image was to be cropped to size 150 and we are looking at 6th column which has 50 as the delta, we crop it with a single crop size of 150 - 50 =100.
LSTM:
Logfile                 0           10          20          30      40      50      60      70      80
log_model10_42.txt      13.85       13.87       14.06       14.37   14.91   15.73   16.82   18.17   19.90
log_model10_43.txt      13.93       13.93       14.02       14.34   14.89   15.80   17.11   18.71   20.99

Static + 175 centercrop
Logfile                 0           10          20          30      40      50      60      70      80
log_model9_74.txt       14.13       14.21       14.45       14.90   15.62   16.64   18.46   20.95   24.67
log_model9_81.txt       14.38       14.52       14.72       15.22   15.88   16.92   18.66   21.15   24.96
log_model9_82.txt       14.31       14.42       14.66       15.08   15.83   16.85   18.43   20.67   24.15


## Look at numbers on the basis of servers. It should not be the case that there is some server which is worsening the results.

## Detoriation in angular error with increase in magnification.
Here, instead of taking original 224*224 sized image, we take that original image and crop it with 224 - delta size and
rescale it to 224*224.Performance is shown with various deltas.
log_model10_42
[ImageLoader] reverse:False random_walk:False forward_bias:0.7 Time:False SeqLen:7 ImgSize:224
[ImageLoaderSineCosineMultiSizedCrops] Cropsizes:[144, 128, 112, 96, 112, 128, 144]
80 tensor(17.3699)
[ImageLoader] reverse:False random_walk:False forward_bias:0.7 Time:False SeqLen:7 ImgSize:224
[ImageLoaderSineCosineMultiSizedCrops] Cropsizes:[154, 137, 120, 103, 120, 137, 154]
70 tensor(16.1800)
[ImageLoader] reverse:False random_walk:False forward_bias:0.7 Time:False SeqLen:7 ImgSize:224
[ImageLoaderSineCosineMultiSizedCrops] Cropsizes:[164, 146, 128, 109, 128, 146, 164]
60 tensor(15.4585)
[ImageLoader] reverse:False random_walk:False forward_bias:0.7 Time:False SeqLen:7 ImgSize:224
[ImageLoaderSineCosineMultiSizedCrops] Cropsizes:[174, 155, 135, 116, 135, 155, 174]
50 tensor(14.8851)
[ImageLoader] reverse:False random_walk:False forward_bias:0.7 Time:False SeqLen:7 ImgSize:224
[ImageLoaderSineCosineMultiSizedCrops] Cropsizes:[184, 164, 143, 123, 143, 164, 184]
40 tensor(14.4666)
[ImageLoader] reverse:False random_walk:False forward_bias:0.7 Time:False SeqLen:7 ImgSize:224
[ImageLoaderSineCosineMultiSizedCrops] Cropsizes:[194, 173, 151, 129, 151, 173, 194]
30 tensor(14.1690)
[ImageLoader] reverse:False random_walk:False forward_bias:0.7 Time:False SeqLen:7 ImgSize:224
[ImageLoaderSineCosineMultiSizedCrops] Cropsizes:[204, 182, 159, 136, 159, 182, 204]
20 tensor(13.9840)
[ImageLoader] reverse:False random_walk:False forward_bias:0.7 Time:False SeqLen:7 ImgSize:224
[ImageLoaderSineCosineMultiSizedCrops] Cropsizes:[214, 191, 167, 143, 167, 191, 214]
10 tensor(13.8783)
[ImageLoader] reverse:False random_walk:False forward_bias:0.7 Time:False SeqLen:7 ImgSize:224
[ImageLoaderSineCosineMultiSizedCrops] Cropsizes:[224, 200, 175, 150, 175, 200, 224]
0 tensor(13.8576)

log_model10_43
[ImageLoader] reverse:False random_walk:False forward_bias:0.7 Time:False SeqLen:7 ImgSize:224
[ImageLoaderSineCosineMultiSizedCrops] Cropsizes:[144, 128, 112, 96, 112, 128, 144]
80 tensor(17.4823)
[ImageLoader] reverse:False random_walk:False forward_bias:0.7 Time:False SeqLen:7 ImgSize:224
[ImageLoaderSineCosineMultiSizedCrops] Cropsizes:[154, 137, 120, 103, 120, 137, 154]
70 tensor(16.2618)
[ImageLoader] reverse:False random_walk:False forward_bias:0.7 Time:False SeqLen:7 ImgSize:224
[ImageLoaderSineCosineMultiSizedCrops] Cropsizes:[164, 146, 128, 109, 128, 146, 164]
60 tensor(15.3696)
[ImageLoader] reverse:False random_walk:False forward_bias:0.7 Time:False SeqLen:7 ImgSize:224
[ImageLoaderSineCosineMultiSizedCrops] Cropsizes:[174, 155, 135, 116, 135, 155, 174]
50 tensor(14.8221)
[ImageLoader] reverse:False random_walk:False forward_bias:0.7 Time:False SeqLen:7 ImgSize:224
[ImageLoaderSineCosineMultiSizedCrops] Cropsizes:[184, 164, 143, 123, 143, 164, 184]
40 tensor(14.3915)
[ImageLoader] reverse:False random_walk:False forward_bias:0.7 Time:False SeqLen:7 ImgSize:224
[ImageLoaderSineCosineMultiSizedCrops] Cropsizes:[194, 173, 151, 129, 151, 173, 194]
30 tensor(14.1508)
[ImageLoader] reverse:False random_walk:False forward_bias:0.7 Time:False SeqLen:7 ImgSize:224
[ImageLoaderSineCosineMultiSizedCrops] Cropsizes:[204, 182, 159, 136, 159, 182, 204]
20 tensor(13.9993)
[ImageLoader] reverse:False random_walk:False forward_bias:0.7 Time:False SeqLen:7 ImgSize:224
[ImageLoaderSineCosineMultiSizedCrops] Cropsizes:[214, 191, 167, 143, 167, 191, 214]
10 tensor(13.9404)
[ImageLoader] reverse:False random_walk:False forward_bias:0.7 Time:False SeqLen:7 ImgSize:224
[ImageLoaderSineCosineMultiSizedCrops] Cropsizes:[224, 200, 175, 150, 175, 200, 224]
0 tensor(13.9296)


log_model9_74
CenterCrop(size=(175, 175))
0 tensor(14.1384)
CenterCrop(size=(167, 167))
10 tensor(14.1683)
CenterCrop(size=(159, 159))
20 tensor(14.3249)
CenterCrop(size=(151, 151))
30 tensor(14.5923)
CenterCrop(size=(143, 143))
40 tensor(15.0179)
CenterCrop(size=(135, 135))
50 tensor(15.6210)
CenterCrop(size=(128, 128))
60 tensor(16.2933)
CenterCrop(size=(120, 120))
70 tensor(17.4780)
CenterCrop(size=(112, 112))
80 tensor(19.1435)

log_model9_81
CenterCrop(size=(175, 175))
0 tensor(14.3859)
CenterCrop(size=(167, 167))
10 tensor(14.4610)
CenterCrop(size=(159, 159))
20 tensor(14.6104)
CenterCrop(size=(151, 151))
30 tensor(14.9099)
CenterCrop(size=(143, 143))
40 tensor(15.3060)
CenterCrop(size=(135, 135))
50 tensor(15.8839)
CenterCrop(size=(128, 128))
60 tensor(16.5028)
CenterCrop(size=(120, 120))
70 tensor(17.6598)
CenterCrop(size=(112, 112))
80 tensor(19.2913)

log_model9_82
CenterCrop(size=(175, 175))
0 tensor(14.3149)
CenterCrop(size=(167, 167))
10 tensor(14.3878)
CenterCrop(size=(159, 159))
20 tensor(14.5421)
CenterCrop(size=(151, 151))
30 tensor(14.8115)
CenterCrop(size=(143, 143))
40 tensor(15.2064)
CenterCrop(size=(135, 135))
50 tensor(15.8389)
CenterCrop(size=(128, 128))
60 tensor(16.5026)
CenterCrop(size=(120, 120))
70 tensor(17.5548)
CenterCrop(size=(112, 112))
80 tensor(18.9686)

log_model9_78
CenterCrop(size=(224, 224))
0 tensor(14.3853)
CenterCrop(size=(214, 214))
10 tensor(14.4252)
CenterCrop(size=(204, 204))
20 tensor(14.5678)
CenterCrop(size=(194, 194))
30 tensor(14.7843)
CenterCrop(size=(184, 184))
40 tensor(15.1451)
CenterCrop(size=(174, 174))
50 tensor(15.7586)
CenterCrop(size=(164, 164))
60 tensor(16.5170)
CenterCrop(size=(154, 154))
70 tensor(17.5638)
CenterCrop(size=(144, 144))
80 tensor(18.9056)

log_model9_77
CenterCrop(size=(224, 224))
0 tensor(14.6612)
CenterCrop(size=(214, 214))
10 tensor(14.7110)
CenterCrop(size=(204, 204))
20 tensor(14.8219)
CenterCrop(size=(194, 194))
30 tensor(15.0253)
CenterCrop(size=(184, 184))
40 tensor(15.3890)
CenterCrop(size=(174, 174))
50 tensor(15.9001)
CenterCrop(size=(164, 164))
60 tensor(16.6198)
CenterCrop(size=(154, 154))
70 tensor(17.5487)
CenterCrop(size=(144, 144))
80 tensor(18.7178)

log_model9_79
CenterCrop(size=(224, 224))
0 tensor(14.4093)
CenterCrop(size=(214, 214))
10 tensor(14.4580)
CenterCrop(size=(204, 204))
20 tensor(14.5772)
CenterCrop(size=(194, 194))
30 tensor(14.7772)
CenterCrop(size=(184, 184))
40 tensor(15.1513)
CenterCrop(size=(174, 174))
50 tensor(15.6581)
CenterCrop(size=(164, 164))
60 tensor(16.4142)
CenterCrop(size=(154, 154))
70 tensor(17.4347)
CenterCrop(size=(144, 144))
80 tensor(18.7580)


## Things to do. July 11
1. ~~Using trained backbone from LSTM, train the head using no cropping. This will show the regularization effect.~~ I don't see any benefit initially.
2. Train the non-symmetric to show the effect of symmetric cropping.
3. Do the better-than-random experiment on all backbones so as to validate that idea as well.
4. Attempt another dataset.
5. 175 cropsize shows that there is overall benefit in doing cropping. However, with lstm, we enhance over it. write it up in the paper.
6. Show that in which domain does the lstm model outperforms the basic model.
7. ~~Name the models.~~
8. With opposite seq, show the zoom in effect.
9. I need to train Sequence models.

### Model Names:
1. Static (Paper)
2. StaticSC( Sine Cosine )
3. StaticSCLstm
4. StaticSCLstmSym

1. SeqLstm (Paper)
2. SeqSCLstm
3. SeqSCLstmSym

log_model10_211.txt: nn.LSTM(self.img_feature_dim, self.img_feature_dim, bidirectional=False, num_layers=1, batch_first=True)

## Things to do. July 24
1. Complete the experiments on Gaze360:
        a. Static: Effect of regularization. (350*3) = 1050
        b. Static: Effect of symmetricity with cropsize: [224,200,175] symmetric 2 runs = 300*2 = 600
        c. Static, Seq: Effect of Ordering.: opposite crop.
        d. Seq: resnet18: Without crop, with crop = 450*2 = 900
        e. ~~Seq: Hardnet: With crop.~~
2. Complete the writeup.
3. Try on new dataset.

## Experiments:
1. Regularization helps.
2. Symm + crop helps. 2 figures.
3. Ordering of the cropsize.

## Idea:
1. Investigate on which examples does SeqSCLstmSym outperforms SeqSCLstm and vice versa.
2. Incorporate the importance of what is done for sequence models.
3. Eye crops need high resolution images. which may not be present.
4. We don't need warping. That was done on eye images. For full face that is not applicable since unlike eyes, face is not small enough to be considered as a planar object.

## checklist to debug RTGENE:
1. ~~check that any random index corresponds to correct gaze and image tuple.~~
2. ~~check that all instances are unique.~~
3. ~~check that there is contiguous range of fixed sizes for each person.~~
4. ~~add prints in classes~~
5. ~~check individual classes code.~~
6. ~~check run.py code.~~

## Experiments on RTGENE:
1. Inpainted: With PINBAL loss, I see similar numbers when trained till 100 for static and lstm based model.
2. Inpainted: with mse loss, loss used in the paper, I see numbers improve when early stop with 10 patience is done for static.
3. Inpainted: Better than random experiment: I see that when I do the better than random experiment, where we take the minimum angular error achieved from multiple crop sizes, there is very little improvement. Minimum is obtained at 210 size. So there is not much scope for LSTM based fusion as there aren't different scales present in the dataset.
4. run LSTM with 224,210,224.
5. Inpainted large sized image: doing better than random experiment: no benefit shown. One thing to note is that 215 gave much better validation set than 224. This improvement was larger as compared to the Inpainted with small sized image.
6. Original large sized image: Testing was much more beneficial. Use of original image is recommended.


## Check for sanity
1. ~~Do the 3 fold division is correct?~~
    a. division in code are correct.
    b. when testing and training, train validation is correct.

2. ~~does data loader loads what is asked to load.~~
    a. Code looks good.
    b. code run is good.
3. ~~is the computation of loss correct.~~
4. ~~Are we loading all the data for testing.~~
5. ~~Are there better state of the art results for RT-GENE?~~
