
PPO算法是强化学习算法中目前应用最广的算法，虽然这个算法是2017年发表的，但是至今在整个AI领域下的agent子领域中这个算法都是最主要的强化学习算法（至少目前还没有之一），这个算法尤其在ChatGPT和人形机器人中起到了关键性的作用，可以说PPO算法是当前AI领域最为重要的算法之一（这个可以有之一，比如还有transformer等算法）。


下面给出NVIDIA公司和Google公司分别发布的PPO算法的实现：


NVIDIA公司的PPO算法实现源码地址：


[https://openi.pcl.ac.cn/devilmaycry812839668/Isaac\_rl\_pytorch](https://github.com)


Google公司的PPO算法实现的源码地址：


[https://openi.pcl.ac.cn/devilmaycry812839668/google\_brax\_ppo\_pytorch](https://github.com)


因为PPO算法的论文是公开发表的，因此所有公司对于PPO算法的实现的核心基本都是一致的，但是由于所有公司都是根据原始论文自己重新编写的，因此不同的实现会导致一些细节上的不同，而细节上的不同是有可能导致算法性能上的表现有差异的，因此本文就以NVIDIA公司和Google公司的不同实现上来探究一下这种细节上的差距是否会影响算法的最终性能有较大变化。


为了便于分析，在[https://openi.pcl.ac.cn/devilmaycry812839668/google\_brax\_ppo\_pytorch](https://github.com):[FlowerCloud机场](https://hanlianfangzhi.com)中将NVIDIA公司的实现所用的trick形成了ppo\_nvidia.py，而Google公司的实现细节形成了ppo\_google.py，从而进行性能比较。


可以看到二者实现的主要区别在于loss函数中的critic的loss以及actor的advantage的计算部分，而在这里可以用两个函数的不同实现来表现，具体如下：


Google公司的实现：



```


|  | @torch.jit.export |
| --- | --- |
|  | def compute_gae(self, truncation, termination, reward, values, |
|  | bootstrap_value): |
|  | truncation_mask = 1 - truncation |
|  | # Append bootstrapped value to get [v1, ..., v_t+1] |
|  | values_t_plus_1 = torch.cat( |
|  | [values[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0) |
|  | deltas = reward + self.discounting * ( |
|  | 1 - termination) * values_t_plus_1 - values |
|  | deltas *= truncation_mask |
|  |  |
|  | acc = torch.zeros_like(bootstrap_value) |
|  | vs_minus_v_xs = torch.zeros_like(truncation_mask) |
|  |  |
|  | for ti in range(truncation_mask.shape[0]): |
|  | ti = truncation_mask.shape[0] - ti - 1 |
|  | acc = deltas[ti] + self.discounting * ( |
|  | 1 - termination[ti]) * truncation_mask[ti] * self.lambda_ * acc |
|  | vs_minus_v_xs[ti] = acc |
|  |  |
|  | # Add V(x_s) to get v_s. |
|  | vs = vs_minus_v_xs + values |
|  | vs_t_plus_1 = torch.cat([vs[1:], torch.unsqueeze(bootstrap_value, 0)], 0) |
|  | advantages = (reward + self.discounting * |
|  | (1 - termination) * vs_t_plus_1 - values) * truncation_mask |
|  | return vs, advantages |
|  |  |
|  | @torch.jit.export |
|  | def loss(self, td: Dict[str, torch.Tensor]): |
|  | observation = self.normalize(td['observation']) |
|  | policy_logits = self.policy(observation[:-1]) |
|  | baseline = self.value(observation) |
|  | baseline = torch.squeeze(baseline, dim=-1) |
|  |  |
|  | # Use last baseline value (from the value function) to bootstrap. |
|  | bootstrap_value = baseline[-1] |
|  | baseline = baseline[:-1] |
|  | reward = td['reward'] * self.reward_scaling |
|  | termination = td['done'] * (1 - td['truncation']) |
|  |  |
|  | loc, scale = self.dist_create(td['logits']) |
|  | behaviour_action_log_probs = self.dist_log_prob(loc, scale, td['action']) |
|  | loc, scale = self.dist_create(policy_logits) |
|  | target_action_log_probs = self.dist_log_prob(loc, scale, td['action']) |
|  |  |
|  | with torch.no_grad(): |
|  | vs, advantages = self.compute_gae( |
|  | truncation=td['truncation'], |
|  | termination=termination, |
|  | reward=reward, |
|  | values=baseline, |
|  | bootstrap_value=bootstrap_value) |
|  |  |
|  | rho_s = torch.exp(target_action_log_probs - behaviour_action_log_probs) |
|  | surrogate_loss1 = rho_s * advantages |
|  | surrogate_loss2 = rho_s.clip(1 - self.epsilon, |
|  | 1 + self.epsilon) * advantages |
|  | policy_loss = -torch.mean(torch.minimum(surrogate_loss1, surrogate_loss2)) |
|  |  |
|  | # Value function loss |
|  | v_error = vs - baseline |
|  | v_loss = torch.mean(v_error * v_error) * 0.5 * 0.5 |
|  |  |
|  | # Entropy reward |
|  | entropy = torch.mean(self.dist_entropy(loc, scale)) |
|  | entropy_loss = self.entropy_cost * -entropy |
|  |  |
|  | return policy_loss + v_loss + entropy_loss |
|  |  |


```

nvidia公司的实现：



```


|  | @torch.jit.export |
| --- | --- |
|  | def compute_gae(self, truncation, termination, reward, values, |
|  | bootstrap_value): |
|  | truncation_mask = 1 - truncation |
|  | # Append bootstrapped value to get [v1, ..., v_t+1] |
|  | values_t_plus_1 = torch.cat( |
|  | [values[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0) |
|  | deltas = reward + self.discounting * ( |
|  | 1 - termination) * values_t_plus_1 - values |
|  | deltas *= truncation_mask |
|  |  |
|  | acc = torch.zeros_like(bootstrap_value) |
|  | vs_minus_v_xs = torch.zeros_like(truncation_mask) |
|  |  |
|  | for ti in range(truncation_mask.shape[0]): |
|  | ti = truncation_mask.shape[0] - ti - 1 |
|  | acc = deltas[ti] + self.discounting * ( |
|  | 1 - termination[ti]) * truncation_mask[ti] * self.lambda_ * acc |
|  | vs_minus_v_xs[ti] = acc |
|  |  |
|  | # Add V(x_s) to get v_s. |
|  | vs = vs_minus_v_xs + values |
|  | vs_t_plus_1 = torch.cat([vs[1:], torch.unsqueeze(bootstrap_value, 0)], 0) |
|  | advantages = (reward + self.discounting * |
|  | (1 - termination) * vs_t_plus_1 - values) * truncation_mask |
|  | return vs, advantages |
|  |  |
|  |  |
|  | @torch.jit.export |
|  | def compute_gae_nvidia(self, truncation, termination, reward, values, |
|  | bootstrap_value): |
|  | truncation_mask = 1 - truncation |
|  | # Append bootstrapped value to get [v1, ..., v_t+1] |
|  | values_t_plus_1 = torch.cat( |
|  | [values[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0) |
|  | deltas = reward + self.discounting * ( |
|  | 1 - termination) * values_t_plus_1 - values |
|  | deltas *= truncation_mask |
|  |  |
|  | acc = torch.zeros_like(bootstrap_value) |
|  | vs_minus_v_xs = torch.zeros_like(truncation_mask) |
|  |  |
|  | for ti in range(truncation_mask.shape[0]): |
|  | ti = truncation_mask.shape[0] - ti - 1 |
|  | acc = deltas[ti] + self.discounting * ( |
|  | 1 - termination[ti]) * truncation_mask[ti] * self.lambda_ * acc |
|  | vs_minus_v_xs[ti] = acc |
|  |  |
|  | # Add V(x_s) to get v_s. |
|  | vs = vs_minus_v_xs + values |
|  | # vs_t_plus_1 = torch.cat([vs[1:], torch.unsqueeze(bootstrap_value, 0)], 0) ##### 后修改 |
|  | # advantages = (reward + self.discounting *                                 ##### 后修改 |
|  | #               (1 - termination) * vs_t_plus_1 - values) * truncation_mask |
|  | # return vs, advantages                                                     ##### 后修改 |
|  | return vs, (vs_minus_v_xs - vs_minus_v_xs.mean())/(vs_minus_v_xs.std()+1e-8)##### 后修改 |
|  | return vs, (vs_minus_v_xs - vs_minus_v_xs.mean())/(vs_minus_v_xs.std()+1e-8)* truncation_mask##### 后修改 |
|  |  |
|  |  |
|  | @torch.jit.export |
|  | def loss(self, td: Dict[str, torch.Tensor]): |
|  | observation = self.normalize(td['observation']) |
|  | policy_logits = self.policy(observation[:-1]) |
|  | new_baseline = self.value(observation[:-1])        ##### 后修改 |
|  | new_baseline = torch.squeeze(new_baseline, dim=-1) ##### 后修改 |
|  | # baseline = self.value(observation) |
|  | # baseline = torch.squeeze(baseline, dim=-1) |
|  | baseline = td["value"]                       ##### 后修改 |
|  | baseline = torch.squeeze(baseline, dim=-1)   ##### 后修改 |
|  |  |
|  | # Use last baseline value (from the value function) to bootstrap. |
|  | bootstrap_value = baseline[-1] |
|  | baseline = baseline[:-1] |
|  | reward = td['reward'] * self.reward_scaling |
|  | termination = td['done'] * (1 - td['truncation']) |
|  |  |
|  | loc, scale = self.dist_create(td['logits']) |
|  | behaviour_action_log_probs = self.dist_log_prob(loc, scale, td['action']) |
|  | loc, scale = self.dist_create(policy_logits) |
|  | target_action_log_probs = self.dist_log_prob(loc, scale, td['action']) |
|  |  |
|  | with torch.no_grad(): |
|  | vs, advantages = self.compute_gae( |
|  | truncation=td['truncation'], |
|  | termination=termination, |
|  | reward=reward, |
|  | values=baseline, |
|  | bootstrap_value=bootstrap_value) |
|  |  |
|  | rho_s = torch.exp(target_action_log_probs - behaviour_action_log_probs) |
|  | surrogate_loss1 = rho_s * advantages |
|  | surrogate_loss2 = rho_s.clip(1 - self.epsilon, |
|  | 1 + self.epsilon) * advantages |
|  | policy_loss = -torch.mean(torch.minimum(surrogate_loss1, surrogate_loss2)) |
|  |  |
|  | # Value function loss |
|  | v_error = vs - new_baseline |
|  | v_loss = torch.mean(v_error * v_error) * 0.5 * 0.5 |
|  |  |
|  | # Entropy reward |
|  | entropy = torch.mean(self.dist_entropy(loc, scale)) |
|  | entropy_loss = self.entropy_cost * -entropy |
|  |  |
|  | return policy_loss + v_loss + entropy_loss |


```

可以看到，NVIDIA公司的实现中在actor的advantage的计算部分是严格按照PPO论文中的公式形式结合GAE后所组成的形式，其主要特点就是GAE是使用old policy下的value计算的，而与之对应的Google公司实现的版本中GAE部分是使用new policy下的value进行计算的，而Google公司的这种对advantage的实现方法是不符合PPO论文中的推导的。


Google公司的实现版本在计算出GAE后又将其加回到value中，然后按照TD(0\)的计算公式再次计算，并用最后的计算值作为ppo算法中的advantage值。


在critic的loss计算中，虽然在原始的PPO论文中并没有给出这部分的实现，但是这部分的实现也是各家公司都有各自的具体实现，从Google和NVIDIA公司的实现中区别在于target\_value的实现部分，由于都是critic\_loss\=MSE(target\_value \- value)\*\*2，因此这部分只需要看具体的target\_value的实现即可。而Google公司和NVIDIA公司的这部分实现都是在GAE的实现上加回到计算GAE时所使用的value从而得到target\_value，由此可以看到这部分的实现上的区别和GAE实现的区别是一致的，那就是Google公司采用的事new\_policy下的value，而NVIDIA公司则是按照ppo论文推导中的那样使用的是old\_policy下的value。


通过上面的分析可以看到NVIDIA公司和Google公司在PPO算法的实现核心上有较大的出入，其中NVIDIA公司的实现版本更为贴近于PPO论文中的数学推导，而Google公司中的实现更像是一种写错了的diy代码，但是就如同AI算法中的很多算法都是由于写错后发现还不错，能work，然后才发明出来一样（比如dropout算法等就是写错代码后回溯一下，review一看发现效果更好才出现的），那么Google公司这种在原始论文的数学推导的基础上自己DIY的那种实现，并且这种DIY是没有理论和公式支持的情况下表现如何呢，下面给出各自实现的性能表现：


注意：每一行的最后值越大代表性能越好，也就是reward越大。


NVIDIA公司实现的PPO算法的性能表现：



> (ppo) devil@OMEN:\~/isaacgym/google\_brax\_ppo\_pytorch(ppo)devil@OMEN: /isaacgym/googlebraxppopytorch
> (ppo) devil@OMEN:\~/isaacgym/google\_brax\_ppo\_pytorch(ppo)devil@OMEN: /isaacgym/googlebraxppopytorch python ppo\_nvidia.py
> 
> 
> \-92\.46276 615\.6472 619\.4814 1598\.58 3491\.018 4389\.2173 4720\.664 4956\.7036 5157\.224 5403\.6167 5660\.0483
> \-293\.88397 428\.9017 683\.35486 759\.5667 2508\.6987 3483\.2837 4241\.0996 4866\.8745 5309\.0547 5607\.0405 5807\.251
> \-343\.99326 541\.02106 538\.8536 1511\.9242 2728\.268 3476\.1067 4154\.212 4556\.845 4844\.14 5185\.3545 5527\.5596
> \-191\.4029 666\.20013 568\.6209 1540\.9491 2591\.868 3296\.225 3961\.9253 4730\.8076 5330\.6 5634\.892 5967\.589
> \-311\.39725 475\.02048 477\.9977 1368\.197 2588\.6013 3490\.4133 4133\.569 4719\.386 5203\.3667 5467\.0454 5822\.964
> \-63\.62652 624\.08026 500\.64517 1502\.9352 2627\.319 3303\.9001 3867\.9912 4238\.0215 4681\.1646 5099\.2637 5538\.38
> \-408\.4421 510\.3886 498\.45285 1081\.5658 2229\.9773 3054\.2632 3537\.7908 3872\.1826 4265\.0864 4656\.0996 4997\.206
> \-212\.69945 581\.74786 713\.21924 1095\.143 2852\.1592 3918\.4485 4699\.765 5091\.0083 5394\.695 5612\.1733 5851\.659
> \-324\.99445 463\.03882 515\.9956 1046\.4734 2209\.4084 3184\.328 3614\.8186 4063\.7363 4281\.7144 4665\.837 5020\.642
> \-276\.30496 428\.47794 460\.0709 857\.62274 1759\.2151 2813\.151 3311\.0247 3946\.3518 4774\.614 5690\.9824 6539\.924
> \-306\.20178 517\.5707 476\.00766 1057\.7833 2050\.8884 2862\.6584 3293\.738 4310\.5254 4921\.074 5340\.552 5676\.209
> \-299\.69257 623\.89087 482\.17316 1458\.4841 2388\.513 3250\.9512 3694\.5715 3775\.5378 4847\.8716 5599\.755 5873\.143
> \-125\.3056 654\.55273 705\.49445 1482\.1265 2603\.1406 3075\.0476 3668\.7322 4589\.628 5283\.5356 5741\.815 6166\.7705
> \-285\.0059 549\.20746 876\.36383 1402\.4784 2500\.9507 3047\.863 3459\.8203 3703\.797 4114\.6387 4502\.3013 4766\.3794
> \-241\.96617 512\.50684 555\.71185 912\.9197 2015\.5284 2612\.5881 2849\.5393 3432\.3162 4001\.3625 4579\.4155 5394\.184
> \-68\.229324 453\.25262 615\.454 1037\.2614 2050\.4932 2730\.044 3150\.9194 3691\.9043 4222\.9795 5046\.6445 5490\.1016
> \-287\.40823 668\.20135 584\.20404 1834\.8651 2561\.1052 3072\.583 3335\.0125 3985\.2122 4359\.7812 4571\.1724 4551\.153
> \-394\.80255 500\.0413 408\.36472 1182\.8118 2502\.029 3133\.8757 3633\.9517 3946\.0864 4576\.0903 5148\.457 5726\.9873
> \-271\.14374 496\.1476 357\.79 917\.76013 2121\.5967 2780\.4185 3230\.8884 3570\.609 3860\.94 4307\.6743 4990\.6665
> \-352\.497 \-18\.338879 408\.7898 651\.40247 627\.79315 2473\.3062 3442\.1694 3934\.7588 4397\.5635 4772\.0923 5221\.927


Google公司的PPO算法实现的性能表现：



> (ppo) devil@OMEN:\~/isaacgym/google\_brax\_ppo\_pytorch(ppo)devil@OMEN: /isaacgym/googlebraxppopytorch python ppo\_google.py
> 
> 
> \-151\.92789 675\.2054 843\.941 1671\.0323 2254\.3054 3109\.0151 3578\.2156 4327\.9575 4922\.5435 5312\.915 5528\.675
> 
> 
> \-222\.82672 608\.0872 708\.7845 1256\.1017 2426\.4524 3064\.8662 3305\.1748 3814\.818 4554\.5522 5282\.8154 5659\.039
> \-194\.07735 612\.9101 887\.85455 1451\.6993 2490\.8435 3206\.251 4362\.8076 5255\.4697 5905\.1665 6497\.951 6871\.8833
> \-193\.992 575\.5923 616\.4585 1826\.0192 2767\.8145 3423\.667 3964\.7856 4502\.3784 4883\.922 5245\.124 5498\.1953
> \-203\.13052 738\.5986 935\.9803 2012\.8353 2726\.0715 3160\.4214 3391\.8105 3638\.6821 3938\.2808 4264\.0386 4769\.1836
> \-186\.69662 647\.7069 631\.8334 1169\.1359 2479\.4143 3104\.97 3466\.6614 3906\.9832 4365\.091 4677\.6543 4941\.117
> \-355\.6686 584\.8635 958\.773 1538\.586 2527\.5776 3121\.4744 3555\.5793 3789\.8745 3944\.2214 4143\.7837 4565\.5293
> \-113\.433624 628\.0208 1127\.1516 2064\.5857 2751\.531 3127\.0398 3514\.5688 3901\.7441 4413\.523 4819\.1406 5247\.891
> \-180\.58922 548\.8948 710\.517 1568\.603 2407\.2134 2763\.0454 3030\.4236 3327\.0989 3482\.7922 3661\.5115 3834\.1274
> \-299\.12137 590\.0653 761\.4747 1798\.6235 2725\.6143 3309\.9133 4051\.2483 4577\.1196 5234\.373 5494\.769 5713\.4204
> \-304\.54407 629\.31726 734\.1538 1647\.8328 2612\.3733 3263\.9976 3622\.268 4141\.8755 4711\.332 5183\.674 5509\.048
> \-198\.04155 685\.04913 644\.2389 1482\.0554 2523\.795 3091\.4492 3477\.1665 3695\.65 4109\.457 4345\.8647 4835\.086
> \-279\.81683 763\.9339 884\.2232 1734\.2968 2639\.7998 3131\.6545 3823\.177 4479\.641 5142\.165 5552\.4385 5684\.3203
> \-227\.01794 575\.051 791\.56024 1349\.687 2421\.2747 2967\.627 3403\.47 3918\.3408 4583\.2026 5098\.737 5415\.042
> \-188\.58012 614\.0997 601\.72015 1463\.2482 2654\.8445 3279\.124 3575\.5994 3773\.3477 3757\.1409 4159\.688 4267\.0933
> \-381\.04434 556\.0338 778\.38 1440\.889 2346\.758 2832\.1013 3354\.068 4011\.1665 4585\.723 5066\.0977 5489\.0034
> \-238\.84843 608\.9157 657\.5136 1570\.1979 2383\.2021 2841\.463 3231\.8225 3569\.5278 3804\.6384 4278\.828 4853\.5464
> \-272\.51273 671\.35693 718\.05566 1686\.8368 2702\.0715 3344\.3562 3754\.391 4116\.043 4560\.265 4979\.059 5207\.547
> \-198\.75876 538\.0805 861\.5901 1772\.03 2675\.996 3229\.8008 3636\.4485 4043\.887 4436\.2393 4858\.293 5114\.886
> \-322\.20215 741\.1053 711\.51953 1981\.2828 2622\.7393 3018\.7263 3448\.417 4047\.6506 4599\.3955 5183\.3896 5554\.3145


这里我并没有给出最终的结果的平均和求方差操作，因为在这种比较少的20次重复试验下二者结果在相近的情况下是无法分出谁好谁坏的，因此在有了上面的性能结果对比后我们可以得到下面的几个结论：


1. 在原始PPO论文技术上不严格的按照原始数学形式进行的计算也是有可能做到不影响算法性能的（至少没有明显差异），这在某种层面上也说明当前的AI发展所是在数学基础上构建的，都是也只是做到了借鉴和部分使用数学的程度，这并不是数学学科中的数学公式的推导那样，数学理论在AI领域更多的是用来在一个算法发明后进行一定程度上的解释而很难能够用来推导出AI算法，更难以用来区分哪个AI算法好坏的，或者说目前的AI算法更多的可以被认为是实践派而不是理论派；
2. 虽然很多AI算法在不同的公司、企业、社区、还有各种AI的算法库（library）中实现细节各有不同，甚至有很大差异，并且很多都和原始发表的论文中的原始形式有较大差异，但是这些不同的实现如果被广大的社区、科研领域、企业公司等采用，那么就说明这种差异的实现并没有导致不同实现下的算法在具体表现中有明显的差异，这也可以要一些完美主义者（本人就属于这种）不需要过度的对不同的library中的实现上的一些不同（包括核心过程的不同，也包括一些细节上的trick的不同）过多的计较，因为经验告诉我们这种差异没啥大的性能差异，不过需要注意这些说的这些不同的实现都是经过各大互联网公司和高校科研院所等广泛使用的，而不是你在GitHub上随便找的那种，如果是一个比较陌生的实现方式还是要谨慎的，毕竟这种是真的没经过广泛实践验证的。


**PS：**


虽然各大公司和library的具体实现的不同并不会造成算法具体表现的明显差异，但是我个人还是偏向于使用那种更贴近于原始论文实现的那种实现，因为这样更好理解。


**个人github博客地址：**
[https://devilmaycry812839668\.github.io/](https://github.com "https://devilmaycry812839668.github.io/")


