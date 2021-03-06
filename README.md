## 作业说明

基于阿里天池“阿里妈妈搜索广告转化预测”课程内容进行实现

> ###### 1 背景
>
> 用户输入关键词搜索后，商品列表将会展现给用户，用户点击进入详情⻚，综合决定是否购买。
>
> 如果有 M 个用户进入同一商品详情⻚，其中 N 个购买了该商品，那么该商品的转化率为 N/M
>
> ###### 2 任务
>
> 选手根据若干天的 5 类数据，预测每个 instance_id 的转化率。 包含:
>
> 1. 日常的转化率预估
> 2. 特殊日期的转化率预估
>
> 选手需要根据数据的分布情况判断是哪一类挑战
>
> 使用下面的公式来评价，越小越好。
>
> logloss=−1NN∑i=1(yilog(pi)+(1−yi)log(1−pi))logloss=−1N∑i=1N(yilog(pi)+(1−yi)log(1−pi))
>
> 其中 N 表示测试集样本数量，yi 表示测试集(这里的测试集是指不公开的测试数 据)中第 i 个样本的真实标签，pi 表示第 i 个样本的预估转化率。
>
> ###### 3 数据说明
>
> 共5类数据：
>
> - 基础数据表提供了搜索广告最基本的信息，以及“是否交易”的标记。
> - 广告商品信息、用户信息、上下文信息和店铺信息等4类数据，提供了对转化率预估可能有帮助的辅助信息。
> - 如果一条样本的某个字段为“-1”，表示这个样本的对应字段缺乏数据。
>
> ###### 4 官方论坛对数据字段业务背景的说明
>
> 1. 数据中的各个字段中的 category 都是一个体系，所有 property 是另一个体系
> 2. 每个广告商品都有类目(category)和属性(property)的字段。一个广告商品， 有从大到小的若干个类目，以及没有从属关系的若干个属性。每个广告商品的类目 数量不固定，属性数量也不固定。举例说明，某一个广告商品的 item_category_list 可能是“手机;全面屏手机”;这个广告商品的item_property_list 可能是“白色;支持 LTE;4.7英寸屏幕”。
> 3. predict_category_property 是根据上下文信息(搜索词、用户信息、候选集广告信 息)推测出的类目和属性，其中的每一个预测类目对应于若干个(可能为零个)预 测属性。举例说明，某一个样本的 predict_category_property 可能对应于“手机:白 色,支持LTE,4.7英寸屏幕;水果:红色,球形;公司:-1”，其中 -1 表示预测类目“公司”没有 找到适用于当前广告商品的预测属性。
> 4. predict_category_property，根据用户输入的关键词，然后系统自己预测的类目和 属性，是根据用户搜索作出的排序结果，越靠前概率越大
> 5. item_property_list 不区分优先级
> 6. item_pv_level字段，广告商品被展示次数是该广告展现给所有用户的总次数
> 7. context_page_id字段，广告商品的展示⻚面编号，加过偏移处理
> 8. context_timestamp字段，该广告展示给用户时的时间，是个时间点，不是时间段
> 9. item_category_list字段，有几个等级是不确定的

根据所给数据，运用前面章节实验作业中提供的过程及代码训练模型并对数据进行预测



## 实现说明

- preprocess.py

  > - 功能：从数据源读取数据，进行预处理；建立训练集和测试集
  > - 实现：照搬课程“基于matplotlib数据可视化分析”的数据预处理部分

- analyzing.py

  > - 功能：探索数据的基本特征并可视化展示
  >
  > - 实现：照搬课程“基于matplotlib数据可视化分析”的数据可视化分析部分

- featuring.py

  > - 功能：建立和添加特征，填充缺失值等
  >
  > - 实现：基础为课程“数据探索与数据预处理”的特征工程部分
  >
  > - ==修改的内容==：
  >
  >   - 将 item_category_list 的第一个种类字段纳入特征集合
  >   - 使用“第n天的某一属性在历史上出现总次数”的一些特征
  >
  >     - 按商品和用户分组的属性数量
  >     - 按用户分组的属性数量
  >     - 按已交易用户分组的属性数量
  >     - 按已交易商品分组的属性数量
  >   - 弃用“样本按某个属性分组后每天出现的时间先后顺序”的一些特征
  >
  >      - 其会导致本地运行失败
  >   - 增加一些额外特征
  >   -  将原本的“构造特征二”的提取信息过程转移到特征处理的最后进行，将其运行之前所添加的特征添加进入信息集合中
  >   - 最终有86个特征参与模型训练
  
- modeling.py

  > - 功能：模型训练和验证
  >- 实现：基础为课程“GBDT及其应用”的模型训练及验证部分
  > - ==修改的内容==：
  >   - 尝试训练微调了模型参数但效果不明显

- result.txt

  > 功能：预测结果文本
  >
  > 包含 instance_id 和 predicted_score
