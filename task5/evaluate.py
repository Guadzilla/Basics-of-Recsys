import math

def rec_eval(val_rec_items, val_user_items, trn_user_items):
    print('recall:',Recall(val_rec_items, val_user_items))
    print('precision',Precision(val_rec_items, val_user_items))
    print('coverage',Coverage(val_rec_items, trn_user_items))
    print('Popularity',Popularity(val_rec_items, trn_user_items)) 

# 评价指标
# 推荐系统推荐正确的商品数量占用户实际点击的商品数量
def Recall(Rec_dict, Val_dict):
    '''
    Rec_dict: 推荐算法返回的推荐列表, 形式:{uid: {item1, item2,...}, uid: {item1, item2,...}, ...} 
    Val_dict: 用户实际点击的商品列表, 形式:{uid: {item1, item2,...}, uid: {item1, item2,...}, ...}
    '''
    hit_items = 0
    all_items = 0
    for uid, items in Val_dict.items():
        rel_set = items
        rec_set = Rec_dict[uid]
        for item in rec_set:
            if item in rel_set:
                hit_items += 1
        all_items += len(rel_set)

    return round(hit_items / all_items * 100, 2)

# 推荐系统推荐正确的商品数量占给用户实际推荐的商品数
def Precision(Rec_dict, Val_dict):
    '''
    Rec_dict: 推荐算法返回的推荐列表, 形式:{uid: {item1, item2,...}, uid: {item1, item2,...}, ...} 
    Val_dict: 用户实际点击的商品列表, 形式:{uid: {item1, item2,...}, uid: {item1, item2,...}, ...}
    '''
    hit_items = 0
    all_items = 0
    for uid, items in Val_dict.items():
        rel_set = items
        rec_set = Rec_dict[uid]
        for item in rec_set:
            if item in rel_set:
                hit_items += 1
        all_items += len(rec_set)

    return round(hit_items / all_items * 100, 2)

# 所有被推荐的用户中,推荐的商品数量占这些用户实际被点击的商品数量
def Coverage(Rec_dict, Trn_dict):
    '''
    Rec_dict: 推荐算法返回的推荐列表, 形式:{uid: {item1, item2,...}, uid: {item1, item2,...}, ...} 
    Trn_dict: 训练集用户实际点击的商品列表, 形式:{uid: {item1, item2,...}, uid: {item1, item2,...}, ...}
    '''
    rec_items = set()
    all_items = set()
    for uid in Rec_dict:
        for item in Trn_dict[uid]:
            all_items.add(item)
        for item in Rec_dict[uid]:
            rec_items.add(item)
    return round(len(rec_items) / len(all_items) * 100, 2)

# 使用平均流行度度量新颖度,如果平均流行度很高(即推荐的商品比较热门),说明推荐的新颖度比较低
def Popularity(Rec_dict, Trn_dict):
    '''
    Rec_dict: 推荐算法返回的推荐列表, 形式:{uid: {item1, item2,...}, uid: {item1, item2,...}, ...} 
    Trn_dict: 训练集用户实际点击的商品列表, 形式:{uid: {item1, item2,...}, uid: {item1, item2,...}, ...}
    '''
    pop_items = {}
    for uid in Trn_dict:
        for item in Trn_dict[uid]:
            if item not in pop_items:
                pop_items[item] = 0
            pop_items[item] += 1
    
    pop, num = 0, 0
    for uid in Rec_dict:
        for item in Rec_dict[uid]:
            if item in pop_items:   # 只统计训练集出现过的物品的流行度
                pop += math.log(pop_items[item] + 1) # 物品流行度分布满足长尾分布,取对数可以使得平均值更稳定
                num += 1  
    return round(pop / num, 3)