from pathlib import Path
from html import escape
import json, math, xml.etree.ElementTree as ET
from PIL import ImageFont

OUT = Path(__file__).resolve().parents[1]
COLORS = {
    'logic': ('#eff6ff', '#2563eb'),
    'reg': ('#f3efff', '#7c3aed'),
    'mem': ('#ecfdf5', '#059669'),
    'control': ('#fff7ed', '#c2410c'),
    'note': ('#f1f5f9', '#94a3b8'),
    'port': ('#ffffff', '#475569'),
}
PAGES=[]

class Page:
    def __init__(self, name, title, subtitle, w=2000, h=1250):
        self.name,self.w,self.h=name,w,h
        self.nodes=[]; self.edges=[]; self.texts=[]; self.n=0
        self.text(50,43,title,30,'#0f172a',True)
        self.text(50,82,subtitle,17,'#475569')
        self.text(50,h-38,'MNIST / 196 → 64 → 10  •  REVIEW COPY  •  All sequential blocks: common clk [1], synchronous rst [1] (reset wires omitted).',15,'#64748b')
        self.text(w-340,h-38,name,15,'#64748b')
        PAGES.append(self)
    def text(self,x,y,t,size=17,color='#334155',bold=False):
        self.texts.append(dict(x=x,y=y,t=t,size=size,color=color,bold=bold))
    def box(self,id,x,y,w,h,title,lines=(),kind='logic',clock=False,size=17):
        n=dict(id=id,x=x,y=y,w=w,h=h,title=title,lines=list(lines),kind=kind,clock=clock,size=size)
        self.nodes.append(n);return id
    def edge(self,points,label='',color='#475569',at=None,dash=False):
        self.edges.append(dict(points=points,label=label,color=color,at=at,dash=dash))
    def netin(self,x,y,name,color='#c2410c',length=85):
        self.edge([(x-length,y),(x,y)],name,color,at=(x-length/2,y-13))
    def netout(self,x,y,name,color='#c2410c',length=85):
        self.edge([(x,y),(x+length,y)],name,color,at=(x+length/2,y-13))
    def note(self,x,y,w,h,title,lines):
        return self.box('note'+str(len(self.nodes)),x,y,w,h,title,lines,'note',False,16)

# 01 — Main datapath. Named control nets keep the datapath readable.
p=Page('01 · System datapath','System datapath','Two dedicated FC engines • 16 lanes each • explicit registered handoffs',2200,1260)
p.box('idx',70,165,230,105,'Selected image',['img_idx_r [5]','accepted + clamped on page 02'],'reg',True)
p.box('imgaddr',420,155,330,125,'Image address',['image_addr = img_idx_r × 196','                         + fc1_act_addr'],'logic')
p.box('imagerom',870,155,270,125,'Image ROM',['3,920 × 8 bits','1-cycle synchronous read'],'mem',True)
p.box('zext',1280,155,280,125,'Zero extension',['pixel9 = {0, pixel8}','Nonnegative signed value [9]'])
p.edge([(300,205),(420,205)],'img_idx_r [5]',at=(360,190))
p.edge([(750,205),(870,205)],'image_addr [12]',at=(810,190))
p.edge([(1140,205),(1280,205)],'pixel8 [8]',at=(1210,190))
p.box('fc1',90,405,285,200,'FC layer 1',['N_IN=196; N_OUT=64','LANES=16; BATCHES=4','Local lane weights + biases','Details: pages 04–06'],'logic',True)
p.box('rqbank',530,405,275,200,'Requant + ReLU bank',['64 parallel instances','SHIFT1 = calibrated constant','1-cycle result + valid','Details: page 07'],'logic',True)
p.box('rqnext',965,400,320,215,'Activation capture logic',['Default: next_a1_bank = a1_bank','if requant_valid:','  next_a1_bank = requant_out','All 64 values captured together'])
p.box('rqreg',1450,415,285,180,'Activation bank',['a1_bank [0:63] × 8 bits','512-bit storage','Hold until next valid capture'],'reg',True)
p.edge([(1560,205),(1630,205),(1630,320),(230,320),(230,405)],'pixel9 [9]',at=(930,305))
p.edge([(90,470),(45,470),(45,120),(570,120),(570,155)],'fc1_act_addr [8]',at=(285,108))
p.edge([(375,505),(530,505)],'acc1 [64 × s32]',at=(452,488))
p.edge([(805,505),(965,505)],'requant_out [64 × s8]',at=(885,482))
p.edge([(1285,505),(1450,505)],'next_a1_bank [512]',at=(1367,483))
p.edge([(1590,595),(1590,655),(1125,655),(1125,615)],'a1_bank [512] · hold feedback',at=(1370,642))
p.text(108,390,'start: fc1_start [1]   done: fc1_done [1]',16,'#c2410c')
p.text(550,390,'en: requant_en [1]',16,'#c2410c')
p.text(984,382,'capture: requant_valid [1]',16,'#059669')
p.netout(805,570,'requant_valid [1]',color='#059669',length=105)
p.box('mux',1450,790,320,155,'Activation selection',['selected8 = a1_bank[fc2_act_addr]','act9 = {0, selected8} [9]'])
p.box('delay',1120,795,185,145,'Delay FF',['act9_delay [9]','1 cycle'],'reg',True)
p.box('fc2',660,755,295,230,'FC layer 2',['N_IN=64; N_OUT=10','LANES=16; BATCHES=1','Lanes 0–9 produce scores','Lanes 10–15: zero initialized'],'logic',True)
p.box('argmax',190,780,310,170,'Clocked argmax',['10 signed 32-bit scores','Largest score; smallest tied index','Result + valid aligned'],'logic',True)
p.edge([(1735,505),(1840,505),(1840,720),(1610,720),(1610,790)],'a1_bank [64 × 8]',at=(1705,706))
p.edge([(1450,865),(1305,865)],'act9 [9]',at=(1377,850))
p.edge([(1120,865),(955,865)],'act9_delay [9]',at=(1037,850))
p.edge([(955,935),(1030,935),(1030,1030),(1830,1030),(1830,895),(1770,895)],'fc2_act_addr [6]',at=(1400,1016))
p.edge([(660,865),(500,865)],'acc2 [10 × s32]',at=(580,848))
p.box('pack',1880,805,270,165,'Debug packing',['For c = 0…9:','  score_debug[32×c +: 32]','    = acc2[c]','Wiring only; no register'],size=15)
p.netin(1880,850,'acc2 [10 × s32]',color='#2563eb',length=85)
p.edge([(2015,970),(2015,1020)],'score_debug [320]',at=(2015,1050))
p.text(680,739,'start: fc2_start [1]   done: fc2_done [1]',16,'#c2410c')
p.text(210,764,'en: argmax_en [1]',16,'#c2410c')
p.edge([(190,865),(100,865),(100,1060),(310,1060)],'argmax_digit [4] + argmax_valid [1]',at=(320,1086))
p.note(480,1070,1280,95,'Result handoff → page 02',['argmax_digit feeds both pred_digit capture and the label comparator; both capture on argmax_valid.','acc2 also feeds score_debug through wiring only. Named control nets connect to the controller on page 02.'])
p.note(1690,155,430,125,'Preserved architecture',['FC1 source latency = image ROM (1).','FC2 source latency = delay FF (1).','Each MAC adds its activation FF (1).'])

# 02 — Top controller and externally visible state.
p=Page('02 · Top control','Top controller, input capture & results','One accepted request • one coherent completion edge • explicit hold feedback',2200,1550)
p.box('nextstate',90,180,390,195,'Next-state logic',['Use transition diagram on page 03','Inputs: start, fc1_done, requant_valid,','fc2_done, argmax_valid, current state','Default: next_state = state'],'control')
p.box('state',660,190,210,175,'State FF',['state [4]','reset → IDLE'],'reg',True)
p.box('decode',1040,165,405,220,'Moore output logic',['Defaults: all outputs = 0','busy = state is a processing state','done = (state == DONE)','Start/enable pulses in *_START only','Full output table: page 03'],'control')
p.edge([(480,275),(660,275)],'next_state [4]',at=(570,257))
p.edge([(870,275),(1040,275)],'state [4]',at=(955,257))
p.edge([(955,275),(955,125),(285,125),(285,180)],'state [4] · feedback',at=(625,114))
p.note(1630,165,475,220,'Controller nets to page 01',['fc1_start [1] → FC1','requant_en [1] → requant bank','fc2_start [1] → FC2','argmax_en [1] → argmax','busy [1], done [1] → external ports'])
p.edge([(1445,275),(1630,275)],'6 control bits',at=(1537,258),color='#c2410c')
p.box('accept',80,510,390,145,'Request acceptance',['accept_start = start AND','                  (state == IDLE OR DONE)','start assumed clean, one-cycle'],'control')
p.text(100,490,'start [1], state [4] →',16,'#c2410c')
p.box('idxnext',650,485,445,195,'Image selection / next value',['Default: next_img_idx = img_idx_r','if accept_start:','  next_img_idx = min(img_idx, 19)','Input img_idx [5]; clamp before storing'])
p.box('idxreg',1260,505,230,155,'Image index FF',['img_idx_r [5]','reset = 0'],'reg',True)
p.edge([(470,575),(650,575)],'accept_start [1]',at=(560,557),color='#c2410c')
p.edge([(1095,575),(1260,575)],'next_img_idx [5]',at=(1177,557))
p.edge([(1490,575),(1550,575),(1550,725),(875,725),(875,680)],'img_idx_r [5] · hold feedback',at=(1200,712))
p.box('labelrom',1715,505,335,155,'Label ROM',['20 × 4 bits','1-cycle synchronous read'],'mem',True)
p.edge([(1490,575),(1715,575)],'img_idx_r [5]',at=(1602,557))
p.text(1290,690,'→ image address, page 01',16,'#2563eb')
p.box('cntnext',80,810,490,200,'Cycle counter / next value',['if accept_start: next_count = 0','else if busy: next_count = count + 1','else: next_count = count','Counts busy cycles through completion edge'])
p.box('cntreg',740,835,265,145,'Cycle count FF',['count [16]','reset = 0'],'reg',True)
p.edge([(570,910),(740,910)],'next_count [16]',at=(655,892))
p.edge([(1005,910),(1055,910),(1055,1040),(325,1040),(325,1010)],'count [16] · hold / increment feedback',at=(720,1027))
p.edge([(870,835),(870,775),(1080,775)],'cycle_count [16]',at=(990,759))
p.text(98,790,'accept_start [1], busy [1] →',16,'#c2410c')
p.box('prednext',80,1180,390,170,'Digit capture / next value',['next_pred = pred_digit','if argmax_valid:','  next_pred = argmax_digit'])
p.box('predreg',640,1195,230,145,'Digit FF',['pred_digit [4]','reset = 0'],'reg',True)
p.edge([(470,1270),(640,1270)],'next_pred [4]',at=(555,1253))
p.edge([(870,1270),(930,1270),(930,1420),(275,1420),(275,1350)],'pred_digit [4] · hold feedback',at=(580,1407))
p.netout(870,1230,'pred_digit [4]',length=125)
p.text(100,1160,'argmax_digit [4], argmax_valid [1] →',16,'#059669')
p.box('cmp',1250,805,465,155,'Label comparison',['match = (argmax_digit == label)','Compare the valid argmax output,','before the final digit FF'])
p.edge([(1880,660),(1880,865),(1715,865)],'label [4]',at=(1810,847))
p.netin(1250,870,'argmax_digit [4]',length=165)
p.box('correctnext',1210,1180,415,170,'Correct capture / next value',['next_correct = correct','if argmax_valid:','  next_correct = match'])
p.box('correctreg',1770,1195,270,145,'Correct FF',['correct [1]','reset = 0'],'reg',True)
p.edge([(1480,960),(1480,1100),(1415,1100),(1415,1180)],'match [1]',at=(1540,1086))
p.edge([(1625,1270),(1770,1270)],'next_correct [1]',at=(1697,1251))
p.edge([(2040,1270),(2110,1270),(2110,1420),(1420,1420),(1420,1350)],'correct [1] · hold feedback',at=(1790,1407))
p.netout(2040,1230,'correct [1]',length=105)
p.text(1230,1160,'argmax_valid [1] →',16,'#059669')
p.note(1210,405,870,85,'One completion edge',['In ARGMAX_WAIT, argmax_valid captures pred_digit and correct while the state FF enters DONE.','No delayed-enable FF is needed. Score debug packing: score_debug[32×c +: 32] = acc2[c], c=0…9.'])

# 03 — top Moore machine
p=Page('03 · Top FSM','Top-level Moore FSM','Start pulses last one state cycle • DONE is held until a new accepted start',2200,1320)
topstates=[('IDLE',100,190),('FC1_START',500,190),('FC1_WAIT',900,190),('REQUANT_START',1300,190),('REQUANT_WAIT',1700,190),('DONE',100,625),('ARGMAX_WAIT',500,625),('ARGMAX_START',900,625),('FC2_WAIT',1300,625),('FC2_START',1700,625)]
for s,x,y in topstates:
    busy=0 if s in ('IDLE','DONE') else 1
    vals=[f'busy={busy}   done={int(s=="DONE")}',f'fc1_start={int(s=="FC1_START")}',f'requant_en={int(s=="REQUANT_START")}',f'fc2_start={int(s=="FC2_START")}',f'argmax_en={int(s=="ARGMAX_START")}']
    p.box(s,x,y,285,205,s,vals,'control',size=17)
p.edge([(385,290),(500,290)],'start',at=(442,270),color='#c2410c')
p.edge([(785,290),(900,290)],'always',at=(842,270),color='#c2410c')
p.edge([(1185,290),(1300,290)],'fc1_done',at=(1242,270),color='#c2410c')
p.edge([(1585,290),(1700,290)],'always',at=(1642,270),color='#c2410c')
p.edge([(1840,395),(1840,625)],'requant_valid',at=(1940,515),color='#059669')
p.edge([(1700,725),(1585,725)],'always',at=(1642,705),color='#c2410c')
p.edge([(1300,725),(1185,725)],'fc2_done',at=(1242,705),color='#c2410c')
p.edge([(900,725),(785,725)],'always',at=(842,705),color='#c2410c')
p.edge([(500,725),(385,725)],'argmax_valid',at=(442,698),color='#059669')
p.edge([(240,625),(240,490),(640,490),(640,395)],'start',at=(445,475),color='#c2410c')
p.edge([(40,150),(240,150),(240,190)],'rst',at=(145,135),color='#7c3aed')
for s,x,y in topstates:
    cond={'IDLE':'!start','DONE':'!start','FC1_WAIT':'!fc1_done','REQUANT_WAIT':'!requant_valid','FC2_WAIT':'!fc2_done','ARGMAX_WAIT':'!argmax_valid'}.get(s)
    if cond:
        p.edge([(x+235,y),(x+235,y-45),(x+70,y-45),(x+70,y)],cond,at=(x+150,y-60),color='#64748b')
p.note(90,970,965,175,'State and output contract',['state / next_state: 4 bits. All six outputs are listed in every state.','start is an input; it is accepted only in IDLE or DONE.','Unrecognized state: next_state = IDLE, outputs = 0.','Each *_START unconditionally advances on the next rising edge.'])
p.note(1160,970,935,175,'Result contract',['argmax_valid and argmax_digit are aligned.','The edge leaving ARGMAX_WAIT captures pred_digit and correct.','After that edge: done=1, busy=0, final cycle_count holds.','In DONE, start goes directly to FC1_START; no extra button press.'])

# 04 — FC controller / counters / datapath
p=Page('04 · FC layer','Parameterized FC layer','Contiguous ownership: neuron = lane × B + batch  •  B = ceil(N_OUT / LANES)',2400,1820)
p.box('ns',70,170,350,170,'Next-state logic',['Transitions: page 05','Inputs: start, weights_done,','batches_done, state','Default: next_state = state'],'control',size=16)
p.box('sf',560,190,210,125,'State FF',['state [3]','reset → IDLE'],'reg',True)
p.box('out',910,160,395,190,'Moore output logic',['acc_clear, count_en, acc_en,','capture_valid, busy, done [1 each]','Decoded from state only','Full values on page 05'],'control')
p.edge([(420,250),(560,250)],'next_state [3]',at=(490,234))
p.edge([(770,250),(910,250)],'state [3]',at=(840,234))
p.edge([(840,250),(840,120),(245,120),(245,170)],'state [3] · feedback',at=(560,108))
p.note(1480,155,810,190,'Signal names / parameters',['q = weight_count; B = NUM_BATCHES; lane L is an elaboration index.','QW=max(1, ceil(log2(N_IN+2))); BW=max(1, ceil(log2(B))).','AW=max(1, ceil(log2(N_IN))); WW=max(1, ceil(log2(B×N_IN))).','acc_clear loads bias; count_en advances q; acc_en permits a MAC.','capture_valid is internal, one cycle per completed batch.'])
p.box('qn',70,475,440,215,'Weight count / next value',['if acc_clear: next_q = 0','else if count_en AND q < N_IN+1:','  next_q = q + 1','else: next_q = q','weights_done = (q == N_IN+1)'])
p.box('qr',675,505,210,140,'Weight count FF',['q [QW]','reset = 0'],'reg',True)
p.edge([(510,575),(675,575)],'next_q [QW]',at=(592,557))
p.edge([(885,575),(935,575),(935,750),(290,750),(290,690)],'q [QW] · feedback',at=(640,734))
p.text(95,451,'acc_clear [1], count_en [1] →',16,'#c2410c')
p.box('clamp',1090,500,350,160,'Clamp input index',['i = min(q, N_IN−1)','act_addr = i [AW]','Clamp before narrowing'])
p.edge([(885,575),(1090,575)],'q [QW]',at=(987,557))
p.netout(1440,550,'act_addr [AW]',length=120)
p.box('wn',1610,530,400,155,'Local weight address',['weight_addr = batch × N_IN + i','All lanes receive the same address','Different lane ROM contents'])
p.edge([(1440,620),(1610,620)],'i [AW]',at=(1525,603))
p.text(1630,510,'batch [BW] →',16,'#c2410c')
p.box('bn',70,910,465,215,'Batch count / next value',['if state == IDLE: next_batch = 0','else if capture_valid AND batch < B−1:','  next_batch = batch + 1','else: next_batch = batch','batches_done = (batch == B−1)'])
p.box('br',690,935,225,140,'Batch FF',['batch [BW]','reset = 0'],'reg',True)
p.edge([(535,1005),(690,1005)],'next_batch [BW]',at=(612,986))
p.edge([(915,1005),(980,1005),(980,1200),(305,1200),(305,1125)],'batch [BW] · feedback',at=(640,1184))
p.text(93,887,'state [3], capture_valid [1] →',16,'#c2410c')
p.note(1080,885,410,205,'Batch fanout / local lookup',['bias_addr = batch [BW] → each lane','batch [BW] → weight address logic','batch [BW] → capture indexing','Hold the last batch; never address B.','Do not clear batch on acc_clear.'])
p.edge([(915,1005),(1080,1005)],'batch [BW]',at=(997,986))
p.box('mac',1750,860,455,285,'MAC lane × LANES',['FC1: 16 × (784 × 8 weight ROM)','FC2: 16 × (64 × 8 weight ROM)','Per-lane bias LUT: B × signed 32','Contiguous neuron n = L × B + batch','acc_clear loads this batch’s bias','Internal datapath: page 06'],'logic',True)
p.edge([(2010,605),(2140,605),(2140,785),(1980,785),(1980,860)],'weight_addr [WW]',at=(2070,767))
p.edge([(1490,965),(1750,965)],'bias_addr [BW]',at=(1615,948))
p.netin(1750,1035,'act_data [9]',length=205)
p.text(1770,840,'acc_clear [1], acc_en [1] →',16,'#c2410c')
p.box('capture',1120,1345,580,250,'Capture next-value logic',['Default: next_acc_out = acc_out  (entire array)','For each elaborated lane L = 0…LANES−1:','  n = L × B + batch','  if capture_valid AND n < N_OUT:','    next_acc_out[n] = mac_lane_out[L]','One combinational owner for the next-value array'])
p.box('accbank',1950,1390,330,165,'Output bank FF',['acc_out [0:N_OUT−1]','N_OUT × signed 32 bits','reset all elements = 0'],'reg',True)
p.edge([(1980,1145),(1980,1240),(1450,1240),(1450,1345)],'mac_lane_out [LANES × s32]',at=(1710,1221))
p.text(1140,1322,'capture_valid [1], batch [BW] →',16,'#c2410c')
p.edge([(1700,1480),(1950,1480)],'next_acc_out [N_OUT × s32]',at=(1825,1459))
p.edge([(2280,1480),(2340,1480),(2340,1690),(1450,1690),(1450,1595)],'acc_out [N_OUT × s32] · hold feedback',at=(1920,1674))
p.note(70,1335,885,245,'Counter and completion interfaces',['weights_done [1] → next-state logic; batches_done [1] → next-state logic.','busy [1], done [1] → FC-layer ports. acc_out array → top datapath.','Counters update at the closing edge; capture uses the current batch.','CLEAR resets q and loads the current bias. FILL_0/FILL_1 advance q.','The final MAC commits at ACCUMULATE → CAPTURE.','CAPTURE writes the stable sum one edge later, then advances batch.'])

# 05 — full FC Moore output specification
p=Page('05 · FC FSM','FC-layer Moore FSM','Two explicit fill cycles • exactly N_IN MAC updates per batch • 3-bit state register',2200,1320)
fcstates=[('IDLE',90,200,(0,0,0,0,0,0)),('CLEAR',610,200,(1,0,0,0,1,0)),('FILL_0',1130,200,(0,1,0,0,1,0)),('FILL_1',1650,200,(0,1,0,0,1,0)),('DONE',90,700,(0,0,0,0,0,1)),('CAPTURE',870,700,(0,0,0,1,1,0)),('ACCUMULATE',1650,700,(0,1,1,0,1,0))]
for s,x,y,v in fcstates:
    p.box(s,x,y,330,225,s,[f'acc_clear = {v[0]}',f'count_en = {v[1]}',f'acc_en = {v[2]}',f'capture_valid = {v[3]}',f'busy = {v[4]}    done = {v[5]}'],'control')
p.edge([(420,315),(610,315)],'start',at=(515,295),color='#c2410c')
p.edge([(940,315),(1130,315)],'always',at=(1035,295),color='#c2410c')
p.edge([(1460,315),(1650,315)],'always',at=(1555,295),color='#c2410c')
p.edge([(1815,425),(1815,700)],'always',at=(1890,555),color='#c2410c')
p.edge([(1650,815),(1200,815)],'weights_done',at=(1425,795),color='#c2410c')
p.edge([(870,815),(420,815)],'batches_done',at=(645,795),color='#c2410c')
p.edge([(1035,700),(1035,550),(775,550),(775,425)],'!batches_done',at=(890,530),color='#c2410c')
p.edge([(255,700),(255,425)],'always',at=(325,558),color='#c2410c')
p.edge([(1920,925),(1920,1000),(1705,1000),(1705,925)],'!weights_done',at=(1810,1024),color='#64748b')
p.edge([(340,200),(340,150),(155,150),(155,200)],'!start',at=(245,134),color='#64748b')
p.edge([(30,240),(90,240)],'rst',at=(55,220),color='#7c3aed')
p.note(80,1090,970,130,'Count meaning',['CLEAR → FILL_0 sets q=0. FILL_0: q=0; FILL_1: q=1.','ACCUMULATE: q=2…N_IN+1; exactly N_IN products commit.','weights_done = (q == N_IN+1). Address i=min(q,N_IN−1).'])
p.note(1140,1090,980,130,'Batch meaning',['batches_done = (batch == B−1). CAPTURE uses current batch.','The capture edge increments batch only if another batch remains.','DONE is one cycle, then IDLE resets batch for the next invocation.'])

# 06 — lane detail
p=Page('06 · MAC lane','One MAC lane','Bias load → signed multiply-accumulate → held accumulator  •  common broadcast addresses',2100,1320)
p.box('waddr',70,200,220,100,'weight_addr',['[WW]'],'port')
p.box('buf',440,170,245,160,'Address buffer FF',['weight_addr_r [WW]','1 cycle; reset = 0'],'reg',True)
p.box('wrom',840,155,360,190,'Local weight ROM',['B × N_IN entries × signed 8','weight_data = ROM[weight_addr_r]','Synchronous read: 1 cycle','Lane-specific initialization file'],'mem',True)
p.edge([(290,250),(440,250)],'weight_addr [WW]',at=(365,232))
p.edge([(685,250),(840,250)],'weight_addr_r [WW]',at=(762,229))
p.box('act',70,490,220,100,'act_data',['[9], zero-extended'],'port')
p.box('actff',440,450,280,180,'Activation FF',['act_data_r [signed 9]','Range 0…255 (FC1)','Range 0…127 (FC2)','1 cycle; reset = 0'],'reg',True)
p.edge([(290,540),(440,540)],'act_data [9]',at=(365,521))
p.box('bias',70,820,220,100,'bias_addr',['batch [BW]'],'port')
p.box('lut',440,780,370,180,'Bias selector / LUT ROM',['bias_in = bias_rom[bias_addr]','B entries × signed 32','Combinational; no clock','Lane-specific initialization file'],'mem')
p.edge([(290,870),(440,870)],'bias_addr [BW]',at=(365,851))
p.box('nextacc',1060,470,560,320,'Accumulator next-value logic',['if acc_clear:','  next_acc = bias_in','else if acc_en:','  product = signed(weight_data) × signed(act_data_r)','  next_acc = acc_out + sign_extend(product)','else: next_acc = acc_out','Product: signed 17 → signed 32'],size=16)
p.box('accff',1780,530,240,180,'Accumulator FF',['acc_out [signed 32]','reset = 0'],'reg',True)
p.edge([(1200,250),(1375,250),(1375,470)],'weight_data [signed 8]',at=(1480,349))
p.edge([(720,540),(1060,540)],'act_data_r [signed 9]',at=(890,521))
p.edge([(810,870),(950,870),(950,740),(1060,740)],'bias_in [signed 32]',at=(975,893))
p.text(1078,448,'acc_clear [1], acc_en [1] →',17,'#c2410c')
p.edge([(1620,620),(1780,620)],'next_acc [s32]',at=(1700,600))
p.edge([(2020,620),(2060,620),(2060,940),(1340,940),(1340,790)],'acc_out [signed 32] · feedback',at=(1670,925))
p.text(1840,750,'→ mac_lane_out[L]',17,'#2563eb')
p.note(70,1090,960,135,'Memory layout for lane L',['n = L × B + batch; weight_rom[batch×N_IN + i] = W[n][i].','bias_rom[batch] = bias[n]. Invalid n: initialize weight/bias entries to zero.','FC1: depth=784, WW=10, BW=2. FC2: depth=64, WW=6, BW=1.'])
p.note(1130,1090,920,135,'Operand timing contract',['External activation source is registered (1 cycle), then act_data_r (1).','Weight address buffer (1 cycle), then registered weight ROM read (1).','acc_en is low during the two fill cycles; controls are not delayed here.'])

# 07 — new requant module
p=Page('07 · Requant + ReLU','One requant + ReLU module','Constant shift • signed intermediates • one registered result stage • identical bank latency',2250,1220)
p.box('in',60,215,225,140,'acc1_in',['signed 32-bit','FC1 sum including bias'],'port')
p.box('round',440,180,330,205,'Round',['R = 2^(SHIFT1−1)','rounded = acc1_in + R','signed 32-bit arithmetic','For SHIFT1=0: R=0'])
p.box('shift',930,180,330,205,'Arithmetic right shift',['shifted = rounded >>> SHIFT1','Keep signed 32-bit value','SHIFT1: calibrated constant'])
p.box('sat',1420,155,385,255,'Saturate to int8',['if shifted > 127: clamp8 = 127','else if shifted < −128: clamp8 = −128','else: clamp8 = shifted[7:0]','Compare wide; narrow only afterward','clamp8 [signed 8]'],size=16)
p.edge([(285,285),(440,285)],'acc1_in [s32]',at=(362,267))
p.edge([(770,285),(930,285)],'rounded [s32]',at=(850,267))
p.edge([(1260,285),(1420,285)],'shifted [s32]',at=(1340,267))
p.box('relu',1420,560,385,150,'ReLU',['a1_comb = (clamp8 < 0) ? 0 : clamp8','a1_comb [signed 8], range 0…127'],size=16)
p.edge([(1610,410),(1610,560)],'clamp8 [signed 8]',at=(1710,490))
p.box('anext',860,535,365,200,'Output next-value logic',['next_a1 = a1_out','if en: next_a1 = a1_comb','Output holds between requests'])
p.box('areg',350,550,310,180,'Output FF',['a1_out [signed 8]','reset = 0','1-cycle result latency'],'reg',True)
p.edge([(1420,635),(1225,635)],'a1_comb [signed 8]',at=(1322,615))
p.edge([(860,635),(660,635)],'next_a1 [signed 8]',at=(760,615))
p.edge([(350,635),(250,635),(250,810),(1040,810),(1040,735)],'a1_out [signed 8] · hold feedback',at=(620,794))
p.text(890,516,'en [1] →',17,'#c2410c')
p.box('vnext',860,955,340,120,'Valid next-value logic',['next_valid = en [1]'],'control')
p.box('vreg',350,955,310,120,'Valid FF',['valid_out [1]; reset = 0'],'reg',True)
p.edge([(860,1015),(660,1015)],'next_valid [1]',at=(760,997),color='#059669')
p.netout(350,1015,'valid_out [1]',length=-170)
p.note(1410,885,725,200,'Bank and capture contract',['en is a one-cycle pulse. valid_out is en delayed one cycle.','All 64 instances share en, clk, rst and SHIFT1.','Use one representative valid as requant_valid; all must agree.','The top activation bank captures on requant_valid.','Require rounded arithmetic to remain within signed int32.'])
p.text(360,765,'a1_out → top activation capture bank',17,'#2563eb')

# 08 — audit-friendly timing and mapping page
p=Page('08 · Timing & mapping','Timing, mapping & interface reference','Implementation contract for the corrected diagrams • review independently of old RTL',2200,1450)
p.box('timing',70,175,1240,450,'One batch — values during each state cycle',['State              q                address i             MAC at closing edge','CLEAR              old              clamped old           load current bias; reset q','FILL_0             0                0                     disabled','FILL_1             1                1                     disabled','ACCUMULATE         2                2                     input 0','ACCUMULATE         …                …                     …','ACCUMULATE         N_IN−1           N_IN−1                input N_IN−3','ACCUMULATE         N_IN             N_IN−1                input N_IN−2','ACCUMULATE         N_IN+1           N_IN−1                input N_IN−1','CAPTURE            N_IN+1           N_IN−1                store final sum; advance batch'],size=19)
p.note(1450,175,650,215,'Cycle budget',['Per batch: CLEAR(1) + FILL(2) + MAC(N_IN)','                  + CAPTURE(1) = N_IN+4 cycles.','FC1: 4 × 200 = 800 cycles. FC2: 1 × 68 = 68.','Top handshakes, requant and argmax add overhead.','Final total depends on the chosen argmax latency.'])
p.note(1450,440,650,185,'Counter terminal rules',['q ranges 0…N_IN+1; hold at terminal.','weights_done = (q == N_IN+1).','batch ranges 0…B−1; hold at terminal.','batches_done = (batch == B−1).'])
p.box('mapping',70,735,1110,230,'Contiguous lane ownership — use the same n everywhere',['n = lane × B + batch; B = ceil(N_OUT / LANES)','FC1: lane 0 → neurons 0,1,2,3; lane 1 → 4,5,6,7; lane 15 → 60,61,62,63.','Local weight address = batch × N_IN + i; local bias address = batch.','Capture destination = acc_out[n]; valid lane result if n < N_OUT.','FC2: B=1; n=lane; only lanes 0…9 write scores.'],size=18)
p.box('widths',1320,735,780,230,'Widths — FC1 / FC2',['Activation address: 8 / 6; q: 8 / 7.','Batch and bias address: 2 / 1 (guard B=1 width).','Local weight address: 10 / 6; weight data: signed 8.','Activation operand: signed 9 with top bit zero.','Product: signed 17; accumulator and bias: signed 32.'],size=18)
p.note(70,1080,1020,220,'Request / result rules',['Only accept start in IDLE or DONE; ignore start while busy.','Latch min(img_idx,19) on accept_start. Image and label stay associated.','Top done stays high in DONE; FC-layer done is a one-cycle pulse.','argmax_valid captures both pred_digit and correct on the DONE-entry edge.','Counter clears on accepted start, counts busy cycles, then holds.'])
p.note(1190,1080,910,220,'Argmax / debug / reset rules',['argmax compares signed scores; ties choose the smallest index.','Clocked argmax returns digit + valid together; latency must be ≥1 cycle.','If scanning: initialize max from score 0; update only on strict >.','score_debug[32×c +: 32] = acc2[c]; each score interpreted signed.','All state/data/control registers reset to 0 except FSM reset state = IDLE.'])

def svg_for(p):
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{p.w}" height="{p.h}" viewBox="0 0 {p.w} {p.h}">',
    '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="context-stroke"/></marker></defs>',
    f'<rect width="{p.w}" height="{p.h}" fill="#ffffff"/>',
    f'<path d="M50 105 H{p.w-50}" stroke="#cbd5e1" stroke-width="2"/>']
    for e in p.edges:
        pts=e['points']
        if len(set(tuple(x) for x in pts))<2:continue
        points=' '.join(f'{x},{y}' for x,y in pts)
        parts.append(f'<polyline points="{points}" fill="none" stroke="{e["color"]}" stroke-width="2" marker-end="url(#arrow)" stroke-linejoin="round"/>')
    for n in p.nodes:
        x,y,w,h=n['x'],n['y'],n['w'],n['h'];fill,stroke=COLORS[n['kind']]
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        title_y=26 if n['kind']=='note' else 32
        line_y=37 if n['kind']=='note' else 46
        parts.append(f'<text x="{x+18}" y="{y+title_y}" font-family="Segoe UI,Arial,sans-serif" font-size="{n["title_size"]}" font-weight="700" fill="#0f172a">{escape(n["title"])}</text>')
        parts.append(f'<path d="M{x+16} {y+line_y} H{x+w-16}" stroke="{stroke}" stroke-opacity="0.28"/>')
        for i,line in enumerate(n['lines']):
            family='Consolas,monospace' if n['id']=='timing' else 'Segoe UI,Arial,sans-serif'
            parts.append(f'<text x="{x+18}" y="{y+n["body_y"]+i*n["leading"]}" font-family="{family}" font-size="{n["size"]}" fill="#334155" xml:space="preserve">{escape(line)}</text>')
        if n['clock']:
            parts.append(f'<path d="M{x} {y+h-27} l13 9 l-13 9" fill="none" stroke="{stroke}" stroke-width="2"/>')
    # Labels are drawn last with opaque backgrounds, preventing wires through text.
    for e in p.edges:
        if e['label'] and e['at']:
            x,y=e['at'];t=e['label']; est=len(t)*8.2
            parts.append(f'<rect x="{x-est/2-5}" y="{y-17}" width="{est+10}" height="23" rx="4" fill="white"/>')
            parts.append(f'<text x="{x}" y="{y}" text-anchor="middle" font-family="Segoe UI,Arial,sans-serif" font-size="16" fill="{e["color"]}">{escape(t)}</text>')
    for t in p.texts:
        parts.append(f'<text x="{t["x"]}" y="{t["y"]}" font-family="Segoe UI,Arial,sans-serif" font-size="{t["size"]}" fill="{t["color"]}" font-weight="{700 if t["bold"] else 400}">{escape(t["t"])}</text>')
    parts.append('</svg>');return '\n'.join(parts)

def drawio():
    mx=ET.Element('mxfile',host='app.diagrams.net',version='29.5.2',pages=str(len(PAGES)))
    for i,p in enumerate(PAGES):
        dia=ET.SubElement(mx,'diagram',name=p.name,id=f'page-{i+1}')
        model=ET.SubElement(dia,'mxGraphModel',dx=str(p.w),dy=str(p.h),grid='1',gridSize='10',guides='1',tooltips='1',connect='1',arrows='1',fold='1',page='1',pageScale='1',pageWidth=str(p.w),pageHeight=str(p.h),background='#ffffff',math='0',shadow='0')
        root=ET.SubElement(model,'root');ET.SubElement(root,'mxCell',id='0');ET.SubElement(root,'mxCell',id='1',parent='0')
        def cell(id,value,style,x,y,w,h,parent='1'):
            c=ET.SubElement(root,'mxCell',id=id,value=value,style=style,vertex='1',parent=parent);ET.SubElement(c,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),attrib={'as':'geometry'});return c
        for j,n in enumerate(p.nodes):
            fill,stroke=COLORS[n['kind']]
            family='Consolas' if n['id']=='timing' else 'Segoe UI'
            cell(n['id'],'','group;',n['x'],n['y'],n['w'],n['h'])
            style=f'rounded=1;whiteSpace=wrap;html=0;fillColor={fill};strokeColor={stroke};strokeWidth=2;arcSize=8;'
            cell(n['id']+'-bg','',style,0,0,n['w'],n['h'],n['id'])
            title_y=26 if n['kind']=='note' else 32
            line_y=37 if n['kind']=='note' else 46
            cell(n['id']+'-title',n['title'],f'text;html=0;align=left;verticalAlign=top;spacing=0;whiteSpace=nowrap;strokeColor=none;fillColor=none;fontFamily=Segoe UI;fontSize={n["title_size"]};fontStyle=1;fontColor=#0f172a;',18,title_y-n['title_size'],n['w']-36,n['title_size']+8,n['id'])
            cell(n['id']+'-rule','',f'fillColor={stroke};strokeColor=none;opacity=25;',16,line_y,n['w']-32,1,n['id'])
            for li,line in enumerate(n['lines']):
                display_line=line.replace(' ','\u00a0') if n['id']=='timing' else line
                cell(n['id']+f'-body{li}',display_line,f'text;html=0;align=left;verticalAlign=top;spacing=0;whiteSpace=nowrap;strokeColor=none;fillColor=none;fontFamily={family};fontSize={n["size"]};fontColor=#334155;',18,n['body_y']+li*n['leading']-n['size'],n['w']-36,n['size']+8,n['id'])
            if n['clock']:
                cell(n['id']+'-clock','',f'shape=triangle;direction=east;fillColor={fill};strokeColor={stroke};strokeWidth=2;',0,n['h']-27,13,18,n['id'])
        for j,e in enumerate(p.edges):
            pts=e['points']
            if len(set(tuple(x) for x in pts))<2:continue
            edge_style=f'edgeStyle=none;rounded=0;html=1;endArrow=block;endFill=1;strokeColor={e["color"]};strokeWidth=2;'
            attrs={}
            for endpoint,key,anchor in [(pts[0],'source','exit'),(pts[-1],'target','entry')]:
                ex,ey=endpoint
                for node in p.nodes:
                    nx,ny,nw,nh=node['x'],node['y'],node['w'],node['h']
                    on_edge=((ex==nx or ex==nx+nw) and ny<=ey<=ny+nh) or ((ey==ny or ey==ny+nh) and nx<=ex<=nx+nw)
                    if on_edge:
                        attrs[key]=node['id']
                        edge_style+=f'{anchor}X={(ex-nx)/nw};{anchor}Y={(ey-ny)/nh};{anchor}Dx=0;{anchor}Dy=0;{anchor}Perimeter=0;'
                        break
            c=ET.SubElement(root,'mxCell',id=f'e{j}',value='',style=edge_style,edge='1',parent='1',**attrs)
            g=ET.SubElement(c,'mxGeometry',relative='1',attrib={'as':'geometry'})
            ET.SubElement(g,'mxPoint',x=str(pts[0][0]),y=str(pts[0][1]),attrib={'as':'sourcePoint'})
            ET.SubElement(g,'mxPoint',x=str(pts[-1][0]),y=str(pts[-1][1]),attrib={'as':'targetPoint'})
            if len(pts)>2:
                a=ET.SubElement(g,'Array',attrib={'as':'points'})
                for x,y in pts[1:-1]:ET.SubElement(a,'mxPoint',x=str(x),y=str(y))
            if e['label'] and e['at']:
                x,y=e['at']; w=max(90,len(e['label'])*8.5)
                cell(f'el{j}',escape(e['label']),f'text;html=1;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;fillColor=#ffffff;strokeColor=none;fontFamily=Arial;fontSize=16;fontColor={e["color"]};',x-w/2,y-21,w,28)
        for j,t in enumerate(p.texts):
            cell(f't{j}',escape(t['t']),f'text;html=1;align=left;verticalAlign=middle;whiteSpace=wrap;strokeColor=none;fillColor=none;fontFamily=Arial;fontSize={t["size"]};fontStyle={1 if t["bold"] else 0};fontColor={t["color"]};',t['x'],t['y']-t['size'],min(p.w-t['x']-20,len(t['t'])*t['size']*.6+50),t['size']+12)
        for c in root.findall('mxCell'):
            for key in ('id','parent','source','target'):
                value=c.get(key)
                if value is not None and value not in ('0','1'):
                    c.set(key,f'p{i+1}-'+value)
    ET.indent(mx);ET.ElementTree(mx).write(OUT/'MNIST accelerator — reviewed.drawio',encoding='utf-8',xml_declaration=True)

if __name__=='__main__':
    for p in PAGES:
        for n in p.nodes:
            font_path='C:/Windows/Fonts/consola.ttf' if n['id']=='timing' else 'C:/Windows/Fonts/segoeui.ttf'
            n['body_y']=56 if n['kind']=='note' else 68
            n['title_size']=18 if n['kind']=='note' else 20
            while max((ImageFont.truetype(font_path,n['size']).getlength(t) for t in n['lines']),default=0)>n['w']-36 and n['size']>12:
                n['size']-=1
            n['leading']=n['size']+7
            if len(n['lines'])>1:
                n['leading']=min(n['leading'],(n['h']-12-n['body_y'])/(len(n['lines'])-1))
            while ImageFont.truetype('C:/Windows/Fonts/segoeuib.ttf',n['title_size']).getlength(n['title'])>n['w']-36:
                n['title_size']-=1
    drawio()
    for i,p in enumerate(PAGES):
        (OUT/'previews'/f'{i+1:02d}.svg').write_text(svg_for(p),encoding='utf-8')
    (OUT/'source'/'scenes.json').write_text(json.dumps([dict(name=p.name,w=p.w,h=p.h,nodes=p.nodes,edges=p.edges,texts=p.texts) for p in PAGES],ensure_ascii=False),encoding='utf-8')
    options=''.join(f'<option value="{i+1:02d}">{escape(p.name)}</option>' for i,p in enumerate(PAGES))
    html='''<!doctype html><html><head><meta charset="utf-8"><title>MNIST architecture review</title><style>
    *{box-sizing:border-box}body{margin:0;background:#e8edf3;color:#172033;font:15px system-ui}header{position:sticky;top:0;z-index:3;background:#101d33;color:white;padding:17px 25px;display:flex;align-items:center;gap:18px;box-shadow:0 2px 10px #0003}h1{font-size:19px;margin:0;margin-right:auto}select,button,a{font:inherit;padding:8px 12px;border-radius:6px;border:0}button,select{cursor:pointer}header a{color:#c7ddff;text-decoration:none}#sheet{padding:24px;overflow:auto;height:calc(100vh - 74px)}img{display:block;background:white;box-shadow:0 3px 18px #2332;width:100%;height:auto;margin:auto}.hint{font-size:13px;color:#b6c4d9}</style></head><body><header><h1>MNIST accelerator · review copy</h1><select id="page">OPTIONS</select><button onclick="zoom(-.2)">−</button><button onclick="fit()">Fit</button><button onclick="zoom(.2)">+</button><a href="MNIST accelerator — reviewed.drawio" download>Editable draw.io</a></header><main id="sheet"><img id="diagram" src="previews/01.svg"></main><script>let scale=1;let el=document.getElementById('diagram');document.getElementById('page').onchange=e=>{el.src='previews/'+e.target.value+'.svg';fit()};function zoom(d){scale=Math.max(.4,Math.min(3,scale+d));el.style.width=(scale*100)+'%';el.style.maxWidth='none'}function fit(){scale=1;el.style.width='100%'}document.addEventListener('keydown',e=>{if(e.key==='ArrowRight'||e.key==='ArrowLeft'){let s=document.getElementById('page');s.selectedIndex=Math.max(0,Math.min(7,s.selectedIndex+(e.key==='ArrowRight'?1:-1)));s.onchange({target:s})}})</script></body></html>'''.replace('OPTIONS',options)
    (OUT/'Preview.html').write_text(html,encoding='utf-8')
    print(f'Created {len(PAGES)} pages in {OUT}')
