## 数据集划分

### 精排

**数据集：**

* 训练集：train_set中前三周

* 验证集：train_set中后三周

* 测试集：test_set

  * session结束日期：

    {1: 219845, 2: 199814, 3: 223385, 4: 267161, 28: 6902, 29: 244627, 30: 260206, 31: 249863}

  * session结束时间：

    {0: 10342, 1: 6957, 2: 6678, 3: 9862, 4: 19830, 5: 37305, 6: 55632, 7: 73070, 8: 80453, 9: 84646, 10: 87527, 11: 91916, 12: 93753, 13: 96050, 14: 101612, 15: 102131, 16: 104070, 17: 115483, 18: 126113, 19: 136348, 20: 112748, 21: 66551, 22: 34753, 23: 17973}


**标签：**观察test中最后一次active的时间分布判断

* 以周为单位截断，选择阶段后的最后一次点击作为标签
* 以周为单位截断，再在一周内任意时间随机截断（日期等概率，小时加权概率），截断后第一次点击作为点击标签；将截断后20个cart和order作为标签

负样本构造：采样的方式构造负样本

**特征：**

* 点击后下单率、加购后下单率、总点击数、总加购数、总下单数、复购率、复加购率
* user_id，item_id，item_graph_embedding
* 总行为序列
* 加购行为序列、下单行为序列（存在同时下单多个item的情况）（avg_pooling）
