import matplotlib.pylab as plt
import seaborn as sns
import datetime
import time
import numpy as np
from sklearn.metrics import accuracy_score
import tensorflow as tf
import util
import load_data
import network

class Project():
    def __init__(self):
        # Setting
        self.labelN = 10
        self.lbl_n = 10
        self.lr = 1e-3
        self.batch_size = 2 * self.labelN * self.lbl_n
        self.Epochs = 20
        self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        # Load Data & Split
        self.sup_img, self.sup_lbl, \
        self.unsup_img, _, \
        self.val_img, self.val_lbl, \
        self.test_img, self.test_lbl = load_data.load_mnist(self.labelN, self.lbl_n)

        # Build Model
        self.model = network.model(self.labelN, self.labelN * self.lbl_n)
        self.op = tf.optimizers.Adam(self.lr)

        # Create writer
        self.checkpoint_dir = f'./checkpoints/{self.current_time}'
        self.train_writer = tf.summary.create_file_writer(f'./logs/{self.current_time}/train')
        self.valid_writer = tf.summary.create_file_writer(f'./logs/{self.current_time}/valid')
        self.checkpoint = tf.train.Checkpoint(optimizer=self.op, model=self.model)
        self.ckpt_manager = tf.train.CheckpointManager(self.checkpoint, directory=self.checkpoint_dir, max_to_keep=2)

    def cclp_loss(self, sup_lbl_onehot, feat, cclp_steps=3):
        # Perform Label Propagation (LP)
        phi_L = sup_lbl_onehot
        # Graph's affinity mtx.
        aff_mtx = tf.matmul(feat, tf.transpose(feat)) / tf.matmul(tf.norm(feat, axis=1, keepdims=True),
                                                                  tf.norm(tf.transpose(feat), axis=0,
                                                                          keepdims=True))
        H_trans = tf.math.exp(aff_mtx) / tf.reshape(tf.reduce_sum(tf.math.exp(aff_mtx), axis=1), [self.batch_size, 1])
        n_split = self.labelN * self.lbl_n
        H_uu = H_trans[n_split:, n_split:]
        H_ul = H_trans[n_split:, :n_split]
        I_uu = tf.eye(num_rows=n_split, num_columns=n_split, dtype=tf.float32)
        Iuu_minus_Huu = tf.subtract(I_uu, H_uu)
        inv_Iuu_minus_Huu = tf.compat.v1.matrix_inverse(Iuu_minus_Huu)
        Hul_mm_Yl = tf.matmul(H_ul, phi_L)
        phi_U = tf.matmul(inv_Iuu_minus_Huu, Hul_mm_Yl)
        phi = tf.concat(values=[phi_L, phi_U], axis=0)

        # Calculate the ideal (target) transition matrix T
        mass_per_c = tf.reduce_sum(phi, axis=0)
        phi_div_mass = phi / (mass_per_c + 1e-5)
        T_trans = tf.matmul(phi, phi_div_mass, transpose_b=True)

        # Compute 1-step CCLP loss
        loss = 0
        H_step = H_trans
        eps_log = 1e-6
        loss_step = tf.reduce_mean(- tf.reduce_sum(T_trans * tf.math.log(H_step + eps_log), axis=[1]))
        loss += loss_step

        # Compute loss over markov chains of multiple length (steps)
        M_class_match = tf.matmul(phi, phi, transpose_b=True)
        H_Masked = H_trans * M_class_match

        for step_i in range(2, cclp_steps + 1):
            H_step = tf.multiply(H_Masked, H_step)
            loss_step = tf.reduce_mean(- tf.reduce_sum(T_trans * tf.math.log(H_step + eps_log), axis=[1]))
            loss += loss_step
        loss /= cclp_steps
        return loss, aff_mtx, H_trans, mass_per_c, T_trans, M_class_match, H_uu, H_ul, Iuu_minus_Huu, inv_Iuu_minus_Huu, Hul_mm_Yl, phi_U

    def pretrain_step(self, images, labels):
        with tf.GradientTape() as tape:
            _, pred = self.model.call(images)
            lbl_onehot = tf.one_hot(labels, self.labelN)
            id_loss = tf.keras.losses.categorical_crossentropy(lbl_onehot, pred)
            id_loss = tf.reduce_sum(id_loss)
        gradients = tape.gradient(id_loss, self.model.trainable_variables)
        self.op.apply_gradients(zip(gradients, self.model.trainable_variables))
        return id_loss

    def train_step(self, sup_img, sup_lbl, unsup_img):
        with tf.GradientTape() as tape:
            sup_feat, sup_pred = self.model.call(sup_img)
            unsup_feat, _ = self.model.call(unsup_img)

            sup_lbl_onehot = tf.one_hot(sup_lbl, self.labelN)
            sup_id_loss = tf.keras.losses.categorical_crossentropy(sup_lbl_onehot, sup_pred)
            sup_id_loss = tf.reduce_sum(sup_id_loss)

            feat = tf.concat([sup_feat, unsup_feat], axis=0)
            cclp_loss, _, H, _, _, _, _, _, _, _, _, phi_U = self.cclp_loss(sup_lbl_onehot, feat)
            total_loss = sup_id_loss + cclp_loss
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.op.apply_gradients(zip(gradients, self.model.trainable_variables))
        return H, sup_id_loss, cclp_loss, phi_U

    def test_step(self, images, labels):
        _, pred = self.model.call(images)

        lbl_onehot = tf.one_hot(labels, self.labelN)
        id_loss = tf.keras.losses.categorical_crossentropy(lbl_onehot, pred)
        id_loss = tf.reduce_sum(id_loss)

        pred = np.argmax(pred, axis=1)
        id_acc = accuracy_score(pred, labels)
        return id_loss, id_acc

    def run(self):
        print('=============================== pretrain ===============================')
        for epoch in range(2):
            start = time.time()
            id_loss_tr = self.pretrain_step(self.sup_img, self.sup_lbl)
            id_loss_val, id_acc_val = self.test_step(self.val_img, self.val_lbl)

            with self.train_writer.as_default():
                tf.summary.scalar(f'ID loss (pretrain)', id_loss_tr, step=epoch + 1)

            with self.valid_writer.as_default():
                tf.summary.scalar(f'ID loss (pretrain)', id_loss_val, step=epoch + 1)
                tf.summary.scalar(f'ID acc (pretrain)', id_acc_val, step=epoch + 1)

            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
            print('Epoch {} | tr_id_loss: {}, val_id_loss:{}, val_id_acc:{}'.format(epoch + 1,
                                                                                     id_loss_tr.numpy(),
                                                                                     id_loss_val.numpy(),
                                                                                     id_acc_val))
        self.ckpt_manager.save(0)

        print('=============================== train ===============================')

        # create paths for saving
        util.mkdir(f'./phi_U')
        util.mkdir(f'./H_mtx')

        for epoch in range(self.Epochs):
            start = time.time()
            L_id = []
            L_cclp = []
            for batch_unsup_img in self.unsup_img:
                H, sup_id_loss, cclp_loss, phi_U = self.train_step(self.sup_img, self.sup_lbl, batch_unsup_img)
                L_id.append(sup_id_loss)
                L_cclp.append(cclp_loss)

            id_loss_val, id_acc_val = self.test_step(self.val_img, self.val_lbl)

            with self.train_writer.as_default():
                tf.summary.scalar(f'ID loss (train)', tf.reduce_mean(L_id), step=epoch + 1)
                tf.summary.scalar(f'CCLP loss (train)', tf.reduce_mean(L_cclp), step=epoch + 1)

            with self.valid_writer.as_default():
                tf.summary.scalar(f'ID loss (train)', id_loss_val, step=epoch + 1)
                tf.summary.scalar(f'ID acc (train)', id_acc_val, step=epoch + 1)

            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
            print('Epoch {} | sup_id_loss: {}, cclp_loss: {}, val_id_loss:{}, val_id_acc:{}'.format(epoch + 1,
                                                                                                    tf.reduce_mean(L_id).numpy(),
                                                                                                    tf.reduce_mean(L_cclp).numpy(),
                                                                                                    id_loss_val.numpy(),
                                                                                                    id_acc_val))
            # save the latest
            sns.heatmap(phi_U)
            plt.savefig(f'./phi_U/epoch' + str(epoch + 1))
            plt.close()

            sns.heatmap(H)
            plt.savefig(f'./H_mtx/H_epoch' + str(epoch + 1))
            plt.close()

        self.ckpt_manager.save(1)

if __name__ == '__main__':
    project = Project()
    project.run()