from tqdm import tqdm
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, trainloader, optimizer, criterion, epochs):
    # criterion = nn.CrossEntropyLoss()
    # criterion.to(device)

    # optimizer = torch.optim.SGD(q1_cnnModel.parameters(), lr=learning_rate
    losses = []
    accuracies = []
        
    for epoch in tqdm(range(epochs)):
        model.train()
        avg_loss = 0
        corect_predictions = 0
        total_predictions = 0

        for i, data in tqdm(enumerate(trainloader), desc=f"  Epoch{epoch} progress : "):
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            # outputs = outputs.to(device)
            loss = criterion(outputs, labels)
            loss.backward()
            avg_loss += loss.item()

            _,pred_lb = outputs.max(dim=1)
            corect_predictions  +=  (pred_lb == labels).sum().item()
            total_predictions+=len(inputs)

            optimizer.step()

        loss = avg_loss / i
        acc = corect_predictions*100.0/total_predictions
        losses.append(loss)
        accuracies.append(acc)
        print(f'    epoch {epoch+1} --> loss: {loss}   acc: {acc}')
    
    print('\nFinished Training')
    
    return {
        "final_loss" : losses[-1],
        "final_accuracy" : accuracies[-1],
        "losses" : losses,
        "accuracies" : accuracies
    }

def test(model, testloader, criterion):
    model.eval()
    # criterion = nn.CrossEntropyLoss()
    # criterion.to(device)

    test_loss = 0
    cor=0
    total=0
    with torch.no_grad():
        for j, (data, label) in enumerate(testloader):
            data = data.to(device)
            label = label.to(device)
            pred = model(data)
            test_loss += criterion(pred, label).item()
            _,pred_lb = pred.max(dim=1)
            cor+=(pred_lb == label).sum().item()
            total+=len(data)
    test_loss /= len(testloader.dataset)
    test_acc = cor*100/total

    print('====> Test loss: {:.4f} acc: {:.4f}'.format(test_loss,test_acc))
    return test_acc