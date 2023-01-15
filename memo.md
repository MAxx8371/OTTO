## 数据
* 每个用户所有行为序列（点击，加购，下单）
* 预测给定行为序列的下一个点击，加购和下单（最多20个）

## 数据探索及特征工程
### session
1. train：12,899,779
2. test: 1,671,803
train和test不重合

### events:
1. train：
<10 events: 0.6391379263164121 
<50 events: 0.9246156077557608 
<150 events: 0.9861470494959642 

1. test：
<10 events: 0.9071541323947857 
<50 events: 0.9947140901170772 
<150 events: 0.9997308295295558

### types of events:
1. train & test:
clicks: 90%
carts:  7.5%
orders: 2.5%
2. unique event count:
	* train:
	70% of sessions have only 1 event type
	17% of sessions have 2 event types
	12% of sessions have all 3 event types
	* test:
	85% of sessions have only 1 event type (much higher than for train)
	12% of sessions have 2 event types
	1% of sessions have all 3 event types (much lower than for train - as number of orders in test is also much lower)

### item
train: 1,855,603 unique products
test: 783,486 unique products
1. test中item都在train中出现过。
2. 训练集和测试集中的热门商品有重合。

### time
* trian：31 July 2022 midnight and goes for 1 month, until 28 Aug 2022
* test：1 week in the future, from 28 Aug (where we left off during train) until 4 Sep 2022


### 样本构造：

#### 召回400个item
使用deep walk进行召回：相隔超过15min则划入下一个session
length = 10
embedding_size = 128

#### 精排
将行为序列以周为单位进行划分；
* user侧特征：

* item侧特征：
  * 被不同session点击、加购、购买的次数
  * 被同一session点击、加购、购买的次数
* 交叉特征：
  * 是否点击、加购、下单过当前item；如果是：几次？最近一次距离目前的时间？多次的平均间隔；