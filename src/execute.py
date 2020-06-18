import torch
from tqdm import tqdm
import os
from captum.attr import LayerIntegratedGradients, TokenReferenceBase, visualization
import time


def train(train_iter, val_iter, model, loss_function, args):
    model.train()
    loss_function.train()

    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.init_lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', verbose=True, patience=3)

    pre_val_loss = 1
    end_signal = 0
    for epoch in range(args.start, args.epoch):

        sumloss = 0
        score_sum = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
        for index, batch in tqdm(enumerate(train_iter, 0)):
            (batch_text, text_lengths), batch_label = batch.Text, batch.Label
            batch_text = batch_text.transpose(0, 1)

            loss, score = loss_function(model(batch_text, text_lengths), batch_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sumloss += loss.item()
            if score:
                for key in score.keys():
                    score_sum[key] += score[key]


        avgloss = sumloss / len(train_iter)
        acc = (score_sum['tp'] + score_sum['tn']) / len(train_iter.dataset)
        precision = score_sum['tp'] / (score_sum['tp'] + score_sum['fp'])
        recall = score_sum['tp'] / (score_sum['tp'] + score_sum['fn'])
        f1 = 2 * precision * recall / (precision + recall)
        print('Epoch %d: train loss, acc and f1:' % epoch, avgloss, acc, f1)

        ckpt = {
            'state_dict': model.state_dict()
        }
        if epoch % 1 == 0:
            torch.save(ckpt, os.path.join(args.checkpoint_dir, "CSA_%s.ckpt" % str(epoch)))

        # scheduler.step(acc)
        val_loss = val(val_iter, model, loss_function, args)
        if pre_val_loss - val_loss < 0: #0.0001:
            end_signal += 1
            if end_signal >= 3:
                print('Overfit warning, end.')
                torch.save(ckpt, os.path.join(args.checkpoint_dir, "CSA_best.ckpt"))
                break
        else:
            end_signal = 0
        print('sub:', pre_val_loss - val_loss)
        pre_val_loss = val_loss


def val(val_iter, model, loss_function, args):
    model.eval()
    loss_function.eval()

    with torch.no_grad():

        sumloss = 0
        score_sum = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
        for index, batch in enumerate(val_iter, 0):
            (batch_text, text_lengths), batch_label = batch.Text, batch.Label
            batch_text = batch_text.transpose(0, 1)

            loss, score = loss_function(model(batch_text, text_lengths), batch_label)

            sumloss += loss.item()
            if score:
                for key in score.keys():
                    score_sum[key] += score[key]

        avgloss = sumloss / len(val_iter)
        acc = (score_sum['tp'] + score_sum['tn']) / len(val_iter.dataset)
        precision = score_sum['tp'] / (score_sum['tp'] + score_sum['fp'])
        recall = score_sum['tp'] / (score_sum['tp'] + score_sum['fn'])
        f1 = 2 * precision * recall / (precision + recall)
        print('val loss, acc and f1:', avgloss, acc, f1)

    model.train()
    loss_function.train()
    return avgloss


def test(test_iter, model, loss_function, args):
    model.eval()
    loss_function.eval()

    os.system('rm ../checkpoints/%s/%s/test.out' % (args.model, args.dataset_name))

    with torch.no_grad():

        sumloss = 0
        score_sum = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
        for index, batch in tqdm(enumerate(test_iter, 0)):
            (batch_text, text_lengths), batch_label = batch.Text, batch.Label
            batch_text = batch_text.transpose(0, 1)

            loss, score = loss_function(model(batch_text, text_lengths), batch_label)

            if args.test_ip and (score['fp'] == 1 or score['fn'] == 1):
                with torch.enable_grad():
                    model.train()  # for calling backward
                    interpret_sentence(model, batch_text, text_lengths, args, batch_label)
                    model.eval()

            sumloss += loss.item()
            if score:
                for key in score.keys():
                    score_sum[key] += score[key]

        avgloss = sumloss / len(test_iter)
        acc = (score_sum['tp'] + score_sum['tn']) / len(test_iter.dataset)
        precision = score_sum['tp'] / (score_sum['tp'] + score_sum['fp'])
        recall = score_sum['tp'] / (score_sum['tp'] + score_sum['fn'])
        f1 = 2 * precision * recall / (precision + recall)
        print('avg loss, acc and f1:', avgloss, acc, f1)


def interpret_sentence(model, text, text_lengths, args, label=0):

    # Interpretable method
    if 'BERT' in args.model:
        PAD_IND = args.bert_tokenizer.pad_token_id
        lig = LayerIntegratedGradients(model, model.model.embeddings)
    else:
        PAD_IND = args.TEXT.vocab.stoi['<pad>']
        lig = LayerIntegratedGradients(model, model.embedding)
    token_reference = TokenReferenceBase(reference_token_idx=PAD_IND)


    model.zero_grad()

    # predict
    start = time.time()
    pred = model(text, text_lengths).squeeze(0)
    print("time:", time.time() - start)
    pred_ind = torch.argmax(pred).item()

    # generate reference indices for each sample
    reference_indices = token_reference.generate_reference(text.shape[1], device=args.device).unsqueeze(0)

    # compute attributions and approximation delta using layer integrated gradients
    attributions_ig_1 = lig.attribute((text, text_lengths), (reference_indices, text_lengths), target=0, n_steps=100,
                                    return_convergence_delta=False)

    attributions_ig_2 = lig.attribute((text, text_lengths), (reference_indices, text_lengths), target=1, n_steps=100,
                                    return_convergence_delta=False)

    if 'BERT' in args.model:
        sentence = [args.bert_tokenizer.ids_to_tokens[int(word)] for word in text.squeeze(0).cpu().numpy() if int(word) != args.bert_tokenizer.pad_token_id]
    else:
        sentence = [args.TEXT.vocab.itos[int(word)] for word in text.squeeze(0).cpu().numpy()]
    # print(sentence)

    add_attributions_to_visualizer(attributions_ig_1, sentence, pred, pred_ind, label, args)
    add_attributions_to_visualizer(attributions_ig_2, sentence, pred, pred_ind, label, args)


def add_attributions_to_visualizer(attributions, text, pred, pred_ind, label, args):
    attributions = attributions.sum(dim=2).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    attributions = attributions.cpu().detach().numpy()

    text_attr = str(
        [(text[i], round(attributions[i], 2)) for i in range(len(text))])  # .replace(", ('<pad>', 0.0)", '')
    if args.interpret:
        print('Pred: (', '%.2f, %.2f' % (pred[0].item(), pred[1].item()), ')', 'attr_sum: ', attributions.sum(),
              text_attr)
    else:
        if pred_ind != label.item():
            with open('../checkpoints/%s/%s/test.out' % (args.model, args.dataset_name), 'a') as f:
                f.write('Pred: (%.2f, %.2f) result_label: %d %d attr_sum: %.2f' % (
                pred[0].item(), pred[1].item(), pred_ind, label.item(), attributions.sum()) + text_attr + '\n')
        print('Pred: (', '%.2f, %.2f' % (pred[0].item(), pred[1].item()), ')', 'result_label: ', pred_ind, label.item(),
              'attr_sum: ', attributions.sum(), text_attr)

