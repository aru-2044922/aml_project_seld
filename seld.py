#
# A wrapper script that trains the network. The training stops when the SELD error stops improving.
#

import os
import tensorflow as tf
import parameter
import sys
import feature_class
import data_generator
import numpy as np
import evaluation_metrics
import time
import matplotlib.pyplot as plt
import crnn_model

# plt.switch_backend('agg')

def collect_test_labels(_data_gen_test, _data_out, quick_test):
    # Collecting ground truth for test data
    nb_batch = 2 if quick_test else _data_gen_test.get_total_batches_in_data()

    batch_size = _data_out[0][0]
    # Create zero matrix of size (16, 128, 11) for SED and (16, 128, 22) for DAO
    gt_sed = np.zeros((nb_batch * batch_size, _data_out[0][1], _data_out[0][2]))
    gt_doa = np.zeros((nb_batch * batch_size, _data_out[0][1], _data_out[1][2]))

    print("nb_batch in test: {}".format(nb_batch))
    cnt = 0
    for tmp_feat, tmp_label in _data_gen_test.generate():
        # tmp_feat -> (16, 128, 64, 7)
        # tmp_label -> (16, 128, 11), (16, 128, 22)
        gt_sed[cnt * batch_size:(cnt + 1) * batch_size, :, :] = tmp_label[0]
        gt_doa[cnt * batch_size:(cnt + 1) * batch_size, :, :] = tmp_label[1]
        cnt = cnt + 1
        if cnt == nb_batch:
            break
    return gt_sed.astype(int), gt_doa

def plot_functions(fig_name, _tr_loss, _val_loss, _sed_loss, _doa_loss, _epoch_metric_loss):
    plt.figure()
    nb_epoch = len(_tr_loss)
    plt.subplot(311)
    plt.plot(range(nb_epoch), _tr_loss, label='train loss')
    plt.plot(range(nb_epoch), _val_loss, label='val loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(312)
    plt.plot(range(nb_epoch), _sed_loss[:, 0], label='sed er')
    plt.plot(range(nb_epoch), _sed_loss[:, 1], label='sed f1')
    plt.plot(range(nb_epoch), _doa_loss[:, 0]/180., label='doa er / 180')
    plt.plot(range(nb_epoch), _doa_loss[:, 1], label='doa fr')
    plt.plot(range(nb_epoch), _epoch_metric_loss, label='seld')
    plt.legend()
    plt.grid(True)

    plt.subplot(313)
    plt.plot(range(nb_epoch), _doa_loss[:, 2], label='pred_pks')
    plt.plot(range(nb_epoch), _doa_loss[:, 3], label='good_pks')
    plt.legend()
    plt.grid(True)

    plt.savefig(fig_name)
    plt.close()


# Function to show the training report for a model
def showTrainingReport(model_name, model_history):
    # sed_out_loss: 0.0418 - doa_out_loss: 0.1694 - val_loss: 22.5391 - val_sed_out_loss: 0.1613 - val_doa_out_loss
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    t = f.suptitle('{}  Perfomance'.format(model_name), fontsize=12)
    f.subplots_adjust(top=0.85, wspace=0.3)

    max_epoch = len(model_history.history['loss'])+1
    epoch_list = list(range(1,max_epoch))
    ax1.plot(epoch_list, model_history.history['sed_out_loss'], label='Train SED Loss')
    ax1.plot(epoch_list, model_history.history['val_{}'.format('sed_out_loss')], label='Validation SED Loss')
    ax1.set_xticks(np.arange(1, max_epoch, 2))
    ax1.set_ylabel('Loss Value')
    ax1.set_xlabel('Epoch')
    ax1.set_title('SED Loss')
    l1 = ax1.legend(loc="best")

    ax2.plot(epoch_list, model_history.history['doa_out_loss'], label='Train DOA Loss')
    ax2.plot(epoch_list, model_history.history['val_{}'.format('doa_out_loss')], label='Validation DOA Loss')
    ax2.set_xticks(np.arange(1, max_epoch, 1))
    ax2.set_ylabel('Loss Value')
    ax2.set_xlabel('Epoch')
    ax2.set_title('DOA Loss')
    l2 = ax2.legend(loc="best")

def main(argv):
    """
    Main wrapper for training sound event localization and detection network.
    
    :param argv: expects two optional inputs. 
        first input: task_id - (optional) To chose the system configuration in parameters.py.
                                (default) 1 - uses default parameters
        second input: job_id - (optional) all the output files will be uniquely represented with this.
                              (default) 1

    """
    if len(argv) != 3:
        print('\n\n')
        print('-------------------------------------------------------------------------------------------------------')
        print('The code expected two optional inputs')
        print('\t>> python seld.py <task-id> <job-id>')
        print('\t\t<task-id> is used to choose the user-defined parameter set from parameter.py')
        print('Using default inputs for now')
        print('\t\t<job-id> is a unique identifier which is used for output filenames (models, training plots). '
              'You can use any number or string for this.')
        print('-------------------------------------------------------------------------------------------------------')
        print('\n\n')

    # use parameter set defined by user
    task_id = '1' if len(argv) < 2 else argv[1]
    params = parameter.get_params(task_id)

    job_id = 1 if len(argv) < 3 else argv[-1]

    train_splits, val_splits, test_splits = None, None, None
    # test_splits = [1, 2, 3, 4]
    # val_splits = [2, 3, 4, 1]
    # train_splits = [[3, 4], [4, 1], [1, 2], [2, 3]]

    # SUGGESTION: Considering the long training time, major tuning of the method can be done on the first split.
    # Once you finlaize the method you can evaluate its performance on the complete cross-validation splits
    test_splits = [1]
    val_splits = [2]
    train_splits = [[3, 4]]

    avg_scores_val = []
    avg_scores_test = []

    for split_cnt, split in enumerate(test_splits):
        print('\n\n---------------------------------------------------------------------------------------------------')
        print('------------------------------------      SPLIT {}   -----------------------------------------------'.format(split))
        print('---------------------------------------------------------------------------------------------------')

        # Unique name for the run
        feature_class.create_folder(params['model_dir'])
        unique_name = '{}_{}_{}_{}_split{}'.format(
            task_id, job_id, params['dataset'], params['mode'], split
        )
        unique_name = os.path.join(params['model_dir'], unique_name)
        model_name = '{}_model.h5'.format(unique_name)

        # Load train and validation data
        # print('Loading training dataset:')
        data_gen_train = data_generator.DataGenerator(
            dataset=params['dataset'], split=train_splits[split_cnt], batch_size=params['batch_size'],
            seq_len=params['sequence_length'], feat_label_dir=params['feat_label_dir']
        )

        # print('Loading validation dataset:')
        data_gen_val = data_generator.DataGenerator(
            dataset=params['dataset'], split=val_splits[split_cnt], batch_size=params['batch_size'],
            seq_len=params['sequence_length'], feat_label_dir=params['feat_label_dir'], shuffle=False
        )

        # Collect the reference labels for validation data
        data_in, data_out = data_gen_train.get_data_sizes()
        print('\nFEATURES:\n\tdata_in: {}\n\tdata_out: {}\n'.format(data_in, data_out))

        gt = collect_test_labels(data_gen_val, data_out, params['quick_test'])
        # gt[0] -> SED -> 32, 128, 11 (32 because it is 2 batches of size 16)
        # gt[1] -> DOA -> 32, 128, 22 (32 because it is 2 batches of size 16)
        sed_gt = evaluation_metrics.reshape_3Dto2D(gt[0]) # SED_GT -> 4096, 11
        doa_gt = evaluation_metrics.reshape_3Dto2D(gt[1]) # DOA_GT -> 4096, 22
        

        # rescaling the reference elevation data from [-180 180] to [-def_elevation def_elevation] for scoring purpose
        nb_classes = data_gen_train.get_nb_classes()
        def_elevation = data_gen_train.get_default_elevation()
        doa_gt[:, nb_classes:] = doa_gt[:, nb_classes:] / (180. / def_elevation)

        print('\nMODEL:\n\tdropout_rate: {}\n\tCNN: nb_cnn_filt: {}, f_pool_size{}, t_pool_size{}\n\trnn_size: {}, fnn_size: {}\n'.format(
                    params['dropout_rate'], params['nb_cnn2d_filt'], params['f_pool_size'], params['t_pool_size'], params['rnn_size'],
                    params['fnn_size']))

        model = crnn_model.get_model(data_in=data_in, data_out=data_out, dropout_rate=params['dropout_rate'],
                                      nb_cnn2d_filt=params['nb_cnn2d_filt'], f_pool_size=params['f_pool_size'], t_pool_size=params['t_pool_size'],
                                      rnn_size=params['rnn_size'], fnn_size=params['fnn_size'],
                                      weights=params['loss_weights'])
        
        best_seld_metric = 99999
        best_epoch = -1
        patience_cnt = 0
        nb_epoch = 2 if params['quick_test'] else params['nb_epochs']
        seld_metric = np.zeros(params['nb_epochs'])
        tr_loss = np.zeros(params['nb_epochs'])
        val_loss = np.zeros(params['nb_epochs'])
        doa_metric = np.zeros((params['nb_epochs'], 6))
        sed_metric = np.zeros((params['nb_epochs'], 2))

        # start training
        print('Started training')
        best_hist = {}
        for epoch_cnt in range(nb_epoch):
            start = time.time()

            hist = model.fit(
                data_gen_train.generate(),
                steps_per_epoch=2 if params['quick_test'] else data_gen_train.get_total_batches_in_data(),
                validation_data=data_gen_val.generate(),
                validation_steps=2 if params['quick_test'] else data_gen_val.get_total_batches_in_data(),
                epochs=params['epochs_per_fit'],
                verbose=1
            )
            tr_loss[epoch_cnt] = hist.history.get('loss')[-1]
            val_loss[epoch_cnt] = hist.history.get('val_loss')[-1]

            # predict once per epoch
            pred = model.predict(
                data_gen_val.generate(),
                steps=2 if params['quick_test'] else data_gen_val.get_total_batches_in_data(),
                verbose=2
            )

            # Calculate the metrics
            # pred[0] --> sed predictions --> 32, 128, 11
            # pred[1] -> DOA -> 32, 128, 22 (32 if it is 2 batches of size 16)
            sed_pred = evaluation_metrics.reshape_3Dto2D(pred[0]) > 0.5  # SED_PRED -> 4096, 11
            doa_pred = evaluation_metrics.reshape_3Dto2D(pred[1]) # DOA_PRED -> 4096, 11

            # rescaling the elevation data from [-180 180] to [-def_elevation def_elevation] for scoring purpose
            # np.concatenate((sed_pred[:int(len(sed_pred)/2), :], sed_gt[int(len(sed_gt)/2):, :]), axis=0))) --> testing
            sed_metric[epoch_cnt, :] = evaluation_metrics.compute_sed_scores(sed_pred, sed_gt, data_gen_val.nb_frames_1s()) #nb_frames_1s --> 50
            doa_metric[epoch_cnt, :] = evaluation_metrics.compute_doa_scores_regr(doa_pred, doa_gt, sed_pred, sed_gt)
            seld_metric[epoch_cnt] = evaluation_metrics.compute_seld_metric(sed_metric[epoch_cnt, :], doa_metric[epoch_cnt, :])

            # Visualize the metrics with respect to epochs
            plot_functions(unique_name, tr_loss, val_loss, sed_metric, doa_metric, seld_metric)

            patience_cnt += 1
            # save model if it outperforms current best model for this cross validation split
            if seld_metric[epoch_cnt] < best_seld_metric:
                best_seld_metric = seld_metric[epoch_cnt]
                best_epoch = epoch_cnt
                model.save(model_name)
                best_hist = hist
                patience_cnt = 0

            print(
                'epoch_cnt: %d, time: %.2fs, tr_loss: %.2f, val_loss: %.2f, '
                'ER_overall: %.2f, F1_overall: %.2f, '
                'doa_error_pred: %.2f, frame_recall:%.2f, '
                'seld_score: %.2f, best_seld_score: %.2f, best_epoch : %d\n' %
                (
                    epoch_cnt + 1, time.time() - start, tr_loss[epoch_cnt], val_loss[epoch_cnt],
                    sed_metric[epoch_cnt, 0], sed_metric[epoch_cnt, 1],
                    doa_metric[epoch_cnt, 0], doa_metric[epoch_cnt, 1],
                    seld_metric[epoch_cnt], best_seld_metric, best_epoch + 1
                )
            )

            # stop training once patience value is exceeded
            if patience_cnt > params['patience']:
                break
        
        # store metrics of best epoch
        avg_scores_val.append([sed_metric[best_epoch, 0], sed_metric[best_epoch, 1], doa_metric[best_epoch, 0],
                               doa_metric[best_epoch, 1], best_seld_metric])

        # Show result for this cross validation split based on the best performing epoch
        print('\nResults on validation split:')
        print('\tUnique_name: {} '.format(unique_name))
        print('\tSaved model for the best_epoch: {}'.format(best_epoch))
        print('\tSELD_score: {}'.format(best_seld_metric))
        print('\tDOA Metrics: DOA_error: {}, frame_recall: {}'.format(doa_metric[best_epoch, 0],
                                                                      doa_metric[best_epoch, 1]))
        print('\tSED Metrics: ER_overall: {}, F1_overall: {}\n'.format(sed_metric[best_epoch, 0],
                                                                       sed_metric[best_epoch, 1]))

        # ------------------  Calculate metric scores for unseen test split ---------------------------------
        print('Loading testing dataset:')
        data_gen_test = data_generator.DataGenerator(
            dataset=params['dataset'], split=split, batch_size=params['batch_size'], seq_len=params['sequence_length'],
            feat_label_dir=params['feat_label_dir'], shuffle=False, per_file=params['dcase_output'],
            is_eval=True if params['mode'] == 'eval' else False
        )

        print('\nLoading the best model and predicting results on the testing split')
        model = tf.keras.models.load_model('{}_model.h5'.format(unique_name))
        pred_test = model.predict_generator(
            generator=data_gen_test.generate(),
            steps=2 if params['quick_test'] else data_gen_test.get_total_batches_in_data(),
            verbose=2
        )

        # Use 0.5 as the threshold for the SED task
        test_sed_pred = evaluation_metrics.reshape_3Dto2D(pred_test[0]) > 0.5
        test_doa_pred = evaluation_metrics.reshape_3Dto2D(pred_test[1])

        # rescaling the elevation data from [-180 180] to [-def_elevation def_elevation] for scoring purpose
        test_doa_pred[:, nb_classes:] = test_doa_pred[:, nb_classes:] / (180. / def_elevation)

        test_data_in, test_data_out = data_gen_test.get_data_sizes()
        test_gt = collect_test_labels(data_gen_test, test_data_out, params['quick_test'])
        test_sed_gt = evaluation_metrics.reshape_3Dto2D(test_gt[0])
        test_doa_gt = evaluation_metrics.reshape_3Dto2D(test_gt[1])
        # rescaling the reference elevation from [-180 180] to [-def_elevation def_elevation] for scoring purpose
        test_doa_gt[:, nb_classes:] = test_doa_gt[:, nb_classes:] / (180. / def_elevation)

        test_sed_loss = evaluation_metrics.compute_sed_scores(test_sed_pred, test_sed_gt, data_gen_test.nb_frames_1s())
        test_doa_loss = evaluation_metrics.compute_doa_scores_regr(test_doa_pred, test_doa_gt, test_sed_pred, test_sed_gt)
        test_metric_loss = evaluation_metrics.compute_seld_metric(test_sed_loss, test_doa_loss)

        avg_scores_test.append([test_sed_loss[0], test_sed_loss[1], test_doa_loss[0], test_doa_loss[1], test_metric_loss])
        print('Results on test split:')
        print('\tSELD_score: {},  '.format(test_metric_loss))
        print('\tDOA Metrics: DOA_error: {}, frame_recall: {}'.format(test_doa_loss[0], test_doa_loss[1]))
        print('\tSED Metrics: ER_overall: {}, F1_overall: {}\n'.format(test_sed_loss[0], test_sed_loss[1]))

        # show summary
        showTrainingReport("CRNN (CNN + Attention + Bidrectional LSTM) Split {}".format(split_cnt + 1), hist)

    print('\n\nValidation split scores per fold:\n')
    for cnt in range(len(val_splits)):
        print('\tSplit {} - SED ER: {} F1: {}; DOA error: {} frame recall: {}; SELD score: {}'.format(cnt, avg_scores_val[cnt][0], avg_scores_val[cnt][1], avg_scores_val[cnt][2], avg_scores_val[cnt][3], avg_scores_val[cnt][4]))

    if params['mode'] == 'dev':
        print('\n\nTesting split scores per fold:\n')
        for cnt in range(len(val_splits)):
            print('\tSplit {} - SED ER: {} F1: {}; DOA error: {} frame recall: {}; SELD score: {}'.format(cnt, avg_scores_test[cnt][0], avg_scores_test[cnt][1], avg_scores_test[cnt][2], avg_scores_test[cnt][3], avg_scores_test[cnt][4]))

if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)