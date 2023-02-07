#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import einops
import fancy_einsum as fe

##% hyperparams
batch_size = 16
context_size = 32
learning_rate = 1e-3
num_epochs = 5000
eval_iters = 100
embedding_dim = 64
num_heads = 4
num_blocks = 4
eval_interval = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# %%
torch.manual_seed(1337)

# open the shakespear_data.txt file and read in
with open("shakespear_data.txt") as f:
    data = f.read()

# make the data a list of characters
chars = sorted(list(set(data)))
vocab_size = len(chars)

# make a dictionary to convert from characters to integers
char_to_int = {ch:i for i,ch in enumerate(chars)}
# make a dictionary to convert from integers to characters
int_to_char = {i:ch for i,ch in enumerate(chars)}
# convert the characters to integers
int_data = [char_to_int[ch] for ch in data]

# make train and test splits
percent_train = 0.9
train_data = int_data[:int(len(int_data)*percent_train)]
val_data = int_data[int(len(int_data)*percent_train):]

# %%
def generate_batch(split):
    # we want to return  2 tensors, x and y
    # x will be the input to the network, y will be the target
    # x will be a tensor of size (batch_size, context_size)
    # y will be a tensor of size (batch_size, context_size)
    # y will be the same as x, but shifted over by one character
    data = train_data if split == 'train' else val_data
    x = torch.zeros((batch_size, context_size), dtype=torch.long)
    y = torch.zeros((batch_size, context_size), dtype=torch.long)
    # get a random starting point for each batch, use torch
    rand_starts = torch.randint(0, len(data)-context_size, (batch_size,))
    # fill in the x and y tensors
    for i, start in enumerate(rand_starts):
        x[i,:] = torch.tensor(data[start:start+context_size])
        y[i,:] = torch.tensor(data[start+1:start+context_size+1])
    return x.to(device), y.to(device)


#%%

# make a function that extimates the loss of the model over epochs
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'test']:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = generate_batch(split)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# %%

class Head(nn.Module):
    """ one head of the attention """

    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(context_size, context_size)))

    def forward(self, x):
        # x is a tensor of size (batch_size, context_length, embedding_dim)
        keys = self.key(x) # (batch_size, context_length, head_size)
        queries = self.query(x) # (batch_size, context_length, head_size)
        values = self.value(x) # (batch_size, context_length, head_size)
        attention = torch.einsum("bkx, bqx->bkq", keys, queries) / x.shape[2]**0.5
        attention = attention.masked_fill(self.mask == 0, float("-inf")) # (batch_size, context_length, context_length)
        attention = F.softmax(attention, dim=-1)
        output = torch.einsum("BTC,BCE->BTE", attention, values)
        return output
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.linear = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(self, x):
        outputs = [head(x) for head in self.heads]
        output = torch.cat(outputs, dim=-1)
        output = self.linear(output)
        return output
    
class FeedForward(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 4*hidden_size),
            nn.ReLU(),
            nn.Linear(4*hidden_size, hidden_size),
        )

    def forward(self, x):
        # x is a tensor of size (batch_size, context_length, embedding_dim)
        # apply the linear layer
        # the output will be a tensor of size (batch_size, context_length, hidden_size)
        output = self.net(x)
        return output

class Block(nn.Module):
    def __init__(self, num_heads, n_embed):
        super().__init__()
        self.attention = MultiHeadAttention(num_heads, n_embed // num_heads)
        self.feed_forward = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        # x is a tensor of size (batch_size, context_length, embedding_dim)
        # apply the attention
        # the output will be a tensor of size (batch_size, context_length, embedding_dim)
        x = x + self.attention(self.ln1(x))
        # apply the feed forward
        # the output will be a tensor of size (batch_size, context_length, embedding_dim)
        x = x + self.feed_forward(self.ln2(x))
        return x

    
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()

        # define the embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.blocks = nn.Sequential(*[Block(num_heads, embedding_dim) for _ in range(num_blocks)])
        self.positional_embedding = nn.Embedding(context_size, embedding_dim)
        self.feed_forward = FeedForward(embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, idx_inputs, idx_targets=None):
        # idx_inputs is a tensor of size (batch_size, context_length)
        # where the values are the indices of the characters
        # the output will be a tensor of size (batch_size, context_length, embedding_dim)
        # idx_targets is a tensor of size (batch_size, context_length)
        # get the embeddings for the inputs
        embed = self.embedding(idx_inputs) # (batch_size, context_length, embedding_dim)
        pos_embed = self.positional_embedding(torch.arange(context_size).to(device)) # (context_length, embedding_dim)
        embed = embed + pos_embed
        logits = self.blocks(embed)
        logits = self.feed_forward(logits)
        logits = self.linear(logits)
        # take the cross entropy. we need to reshape for torch
        # the logits need to be (batch_size, vocab_size, context_length)
        # the targets need to be (batch_size, context_length)
        if idx_targets is None:
            return logits
        else:
            # to compute the loss we need the logits to be (batch_size*context_length, vocab_size)
            # and we need targets to be (batch_size*context_length)
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), idx_targets.view(-1))
            return logits, loss

    def predict(self, idx_inputs, max_steps=100):
        # idx_inputs is a tensor of size (batch_size, context_length)
        # where the values are the indices of the characters
        # the output will be a tensor of size (batch_size, context_length, embedding_dim)
        for i in range(max_steps):
            # get the embeddings for the inputs, of the last context_length characters
            logits = self(idx_inputs[:, -context_size:])
            # get the logits for the last character, which is the prediction
            logits = logits[:,-1,:] # (batch_size, embedding_dim)
            # get the prediction by taking softmax
            probs = F.softmax(logits, dim=1) # (batch_size, vocab_size)
            # get the prediction by sampling from the distribution
            next_char = torch.multinomial(probs, num_samples=1) # (batch_size, 1)
            # add the prediction to the input
            idx_inputs = torch.cat((idx_inputs, next_char), dim=1) # (batch_size, context_length+1)

        return idx_inputs
    
# %%

model = BigramLanguageModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
# train the model, use tqdm to get a progress bar
pbar = tqdm.tqdm(range(num_epochs))
for epoch in pbar:


    # get the data
    idx_inputs, idx_targets = generate_batch('train')
    # move the data to the device
    idx_inputs = idx_inputs.to(device)
    idx_targets = idx_targets.to(device)
    # zero the gradients
    optimizer.zero_grad(set_to_none=True)
    # get the logits
    logits, loss = model(idx_inputs, idx_targets)
    # take the gradient step
    loss.backward()
    optimizer.step()
    # if we are at eval time, print out the loss using estimate, add to the progress bar
    if (epoch) % 100 == 0:
        losses = estimate_loss()
        pbar.set_postfix(loss_train=losses['train'].item(), loss_test=losses['test'].item())
# %%
# make some predictions
idx_inputs, idx_targets = generate_batch('test')
idx_inputs = idx_inputs.to(device)
idx_targets = idx_targets.to(device)
idx_outputs = model.predict(idx_inputs)
# print out the chars of the predictions
for i in range(idx_outputs.shape[0]):
    output = idx_outputs[i,:]
    output = [int_to_char[idx.item()] for idx in output]
    print("".join(output))
# %%
