import numpy as np
import sys

import torch
import torch.nn as nn
import torch.optim as optim


### Define Neural Stack Models

class NNstack(nn.Module):
  def __init__(self, device):
    super().__init__()
    self.device = device

  def forward(self, prev_Val, prev_stg, dt, ut, vt):
    
    # prev_Val.shape = (t-1, ,batch_size, m)
    # prev_stg.shape = (t-1, batch_size)
    # push signal dt is scalar in (0, 1) [batch_szie]
    # pop signal ut is scalar in (0, 1) [batch_size]
    # vt.shape = (1, m)
    t_1, batch_size, m = prev_Val.shape
    # print("batch_size is: ", batch_size)
    # print("m is: ", m)
    t = t_1 + 1
    
    # Update value matrix
    Val = torch.cat((prev_Val, vt), dim=0) #[t,mem_width]
    
    # Update strength vector
    stg = torch.zeros(t, batch_size).to(self.device)
    # print("stg size: ", stg.size())
    # print("dt size: ", dt.size())
    stg[t-1] = dt
    
    for i in np.arange(t_1-1, -1, -1):
      temp = prev_stg[i] - torch.clamp(ut, max=0)
      stg[i] = torch.clamp(temp, max=0)
      ut = ut-stg[i]
      
    # Produce read vector rt.shape = (1, m)
    rt = torch.zeros(batch_size, m).to(self.device)
    
    read = torch.ones((batch_size)).to(self.device)
    for i in np.arange(t-1, -1, -1):
      temp = torch.clamp(read, max=0)
      coef = torch.min(stg[i],temp)
      # print("rt size: ", rt.size())
      # print("Val[i] size: ", Val[i].size())
      rt = rt + torch.unsqueeze(coef,1)*Val[i]
      read = read - stg[i]
    
    return (Val, stg), rt
  
  
class Controller(nn.Module):
  def __init__(self, input_dim, hid_dim, n_layers, device, mem_width=128):
    super().__init__()
    self.device = device
    self.hid_dim = hid_dim
    
    self.mem_width = mem_width
    
    self.nnstack = NNstack(device)
    self.rnn = nn.LSTM(input_dim, hid_dim, n_layers)
    
    self.get_push_signal = nn.Linear(hid_dim, 1, bias=True)
    self.d_sig = nn.Sigmoid()
    self.get_pop_signal = nn.Linear(hid_dim, 1, bias=True)
    self.u_sig = nn.Sigmoid()
    self.get_write_value = nn.Linear(hid_dim, mem_width, bias=True)
    self.v_tan = nn.Tanh()
    self.get_output = nn.Linear(hid_dim, input_dim, bias=True)
    self.o_tan = nn.Tanh()

    # self.Wd = torch.zeros(hid_dim, requires_grad=True).to(self.device)
    # self.Bd = torch.zeros(1, 1, requires_grad=True).to(self.device)
    
    # self.Wu = torch.zeros(hid_dim, requires_grad=True).to(self.device)
    # self.Bu = torch.zeros(1, 1, requires_grad=True).to(self.device)
    
    # self.Wv = torch.zeros(hid_dim, mem_width, requires_grad=True).to(self.device)
    # self.Bv = torch.zeros(1, 1, mem_width, requires_grad=True).to(self.device)
    
    # self.Wo = torch.zeros(hid_dim, input_dim, requires_grad=True).to(self.device)
    # self.Bo = torch.zeros(1, 1, input_dim, requires_grad=True).to(self.device)
    

    self.input_proj = nn.Linear(mem_width+input_dim, input_dim, bias=False)
    nn.init.xavier_normal_(self.input_proj.weight)

  def forward(self, input, prev_State):
    assert input.size()[0] == 1
    batch_size = input.size()[1]
    if prev_State is None:
      prev_Val = torch.zeros(1, batch_size, self.mem_width).to(self.device)
      prev_stg = torch.zeros(1, batch_size).to(self.device)
      prev_hidden_cell = None
      prev_read = torch.zeros(batch_size, self.mem_width).to(self.device)
    else:
      (prev_Val, prev_stg), prev_hidden_cell, prev_read = prev_State
    
    #input and read should be concatenated 128, 128
    input_aug = self.input_proj(torch.cat((input, torch.unsqueeze(prev_read, 0)), dim=2))
    
    if prev_hidden_cell == None:
      output, hidden = self.rnn(input_aug)
    else:
      output, hidden = self.rnn(input_aug, prev_hidden_cell)
    
    # output.shape = [seq_len, 1, hidden_size]

    # print(output)
    dt = self.d_sig(self.get_push_signal(output))
    # dt.shape = (1, 1)
    
    ut = self.u_sig(self.get_pop_signal(output))
    # ut.shape = (1, 1)
    
    vt = self.v_tan(self.get_write_value(output))
    # vt.shape = (1, 1, mem_width)
    
    ot = self.o_tan(self.get_output(output))
    # ot.shape = (1, 1, mem_width)
    
    # print("shape of dt", dt.size())
    # print("shape of ut", ut.size())
    # print("shape of vt", vt.size())
    # print("shape of ot", ot.size())
    dt,ut = torch.squeeze(dt), torch.squeeze(ut)
    (Val, stg), rt = self.nnstack(prev_Val, prev_stg, dt, ut, vt)
    
    State = ((Val, stg), hidden, rt)
    
    return ot, State