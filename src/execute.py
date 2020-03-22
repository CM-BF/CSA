import torch
from tqdm import tqdm
import os
from captum.attr import LayerIntegratedGradients, TokenReferenceBase, visualization
from utils import tokenizer


def train(train_iter, val_iter, model, loss_function, args):

    model.train()
    loss_function.train()

    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.init_lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', verbose=True, patience=3)

    pre_val_loss = 1
    end_signal = 0
    for epoch in range(args.start, args.epoch):

        sumloss = 0
        acc = 0
        for index, batch in tqdm(enumerate(train_iter, 0)):
            batch_text, batch_label = batch.Text.transpose(0, 1), batch.Label

            loss, correct = loss_function(model(batch_text), batch_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sumloss += loss.item()
            if correct:
                acc += correct.item()

        avgloss = sumloss / len(train_iter)
        acc = acc / len(train_iter.dataset)
        print('Epoch %d: train loss and acc:' % epoch, avgloss, acc)

        if epoch % 1 == 0:
            ckpt = {
                'state_dict': model.state_dict()
            }

            torch.save(ckpt, os.path.join(args.checkpoint_dir, "CSA_%s.ckpt" % str(epoch)))
        # scheduler.step(acc)
        val_loss = val(val_iter, model, loss_function, args)
        if pre_val_loss - val_loss < 0.001:
            end_signal += 1
            if end_signal >= 3:
                print('Overfit warning, end.')
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
        acc = 0
        for index, batch in enumerate(val_iter, 0):
            batch_text, batch_label = batch.Text.transpose(0, 1), batch.Label

            loss, correct = loss_function(model(batch_text), batch_label)

            sumloss += loss.item()
            if correct:
                acc += correct.item()

        avgloss = sumloss / len(val_iter)
        acc = acc / len(val_iter.dataset)
        print('val loss and acc:', avgloss, acc)
    return avgloss


def test(test_iter, model, loss_function, args):

    model.eval()
    loss_function.eval()


    os.system('rm test.out')

    with torch.no_grad():

        sumloss = 0
        acc = 0
        for index, batch in tqdm(enumerate(test_iter, 0)):
            batch_text, batch_label = batch.Text.transpose(0, 1), batch.Label

            loss, correct = loss_function(model(batch_text), batch_label)

            if args.test_ip:
                with torch.enable_grad():
                    interpret_sentence(model, batch_text, args, batch_label)

            sumloss += loss.item()
            if correct:
                acc += correct.item()

        avgloss = sumloss / len(test_iter)
        acc = acc / len(test_iter.dataset)
        print('avg loss and acc:', avgloss, acc)


def interpret_sentence(model, text, args, label):

    # Interpretable method
    PAD_IND = args.TEXT.vocab.stoi['<pad>']
    token_reference = TokenReferenceBase(reference_token_idx=PAD_IND)
    lig = LayerIntegratedGradients(model, model.embedding)

    model.zero_grad()

    # predict
    pred = model(text).squeeze(0)
    pred_ind = torch.argmax(pred).item()

    # generate reference indices for each sample
    reference_indices = token_reference.generate_reference(text.shape[1], device=args.device).unsqueeze(0)

    # compute attributions and approximation delta using layer integrated gradients
    attributions_ig = lig.attribute(text, reference_indices, target=1, n_steps=500, return_convergence_delta=False)

    print()#, ', delta: ', abs(delta))

    sentence = [args.TEXT.vocab.itos[int(word)] for word in text.squeeze(0).cpu().numpy()]
    # print(sentence)

    add_attributions_to_visualizer(attributions_ig, sentence, pred, pred_ind, label, args)


def add_attributions_to_visualizer(attributions, text, pred, pred_ind, label, args):
    attributions = attributions.sum(dim=2).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    attributions = attributions.cpu().detach().numpy()

    text_attr = str([(text[i], round(attributions[i], 2)) for i in range(len(text))])#.replace(", ('<pad>', 0.0)", '')
    if args.interpret:
        print('Pred: (', '%.2f, %.2f' % (pred[0].item(), pred[1].item()), ')', 'attr_sum: ', attributions.sum(), text_attr)
    else:
        if pred_ind != label.item():
            with open('test.out', 'a') as f:
                f.write('Pred: (%.2f, %.2f) result_label: %d %d attr_sum: %.2f' % (pred[0].item(), pred[1].item(), pred_ind, label.item(), attributions.sum()) + text_attr + '\n')
        print('Pred: (', '%.2f, %.2f' % (pred[0].item(), pred[1].item()), ')', 'result_label: ', pred_ind, label.item(), 'attr_sum: ', attributions.sum(), text_attr)


def single_interpret(model, text, args, label=torch.tensor(0)):
    model.eval()

    text = tokenizer(text)
    sentence = text
    text = [args.TEXT.vocab.stoi[t] for t in text]
    text = torch.tensor(text, device=args.device).unsqueeze(0)

    # Interpretable method
    PAD_IND = args.TEXT.vocab.stoi['<pad>']
    token_reference = TokenReferenceBase(reference_token_idx=PAD_IND)
    lig = LayerIntegratedGradients(model, model.embedding)

    model.zero_grad()

    # predict
    pred = model(text).squeeze(0)
    pred_ind = torch.argmax(pred).item()

    # generate reference indices for each sample
    reference_indices = token_reference.generate_reference(len(sentence), device=args.device).unsqueeze(0)

    # compute attributions and approximation delta using layer integrated gradients
    attributions_ig = lig.attribute(text, reference_indices, target=1, n_steps=500, return_convergence_delta=False)

    # print(sentence)

    add_attributions_to_visualizer(attributions_ig, sentence, pred, pred_ind, label, args)
