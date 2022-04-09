def calc(n,m,ttuser,ttitem,pre,ttrating,mode,bias,atk=5):
    #ttuser：参与test的userID,ttitem：参与test的itemID
    #pre：prediciton_rating
    #ttrating：GrounTruth
    user=ttuser.cpu().detach().numpy()
    item=ttitem.cpu().detach().numpy()
    pre=pre.cpu().detach().numpy()
    rating=ttrating.cpu().numpy()
    ut,pt,it={},{},{}
    for i,uid in enumerate(user):
        try:
            ut[uid.item()].append(rating[i].item())
            pt[uid.item()].append(pre[i].item())
            it[uid.item()].append(item[i].item())
        except:
            ut[uid.item()]=[rating[i].item()]
            pt[uid.item()]=[pre[i].item()]
            it[uid.item()]=[item[i].item()]
    posuser,positem=[],[]
    for k in ut:
        if (1 in ut[k]) and (-1 in ut[k]):
            for i in range(len(ut[k])):
                if ut[k][i]==1:
                    posuser.append(k)
                    positem.append(it[k][i])

    posuser=np.array(posuser)
    positem=np.array(positem)
    preall=np.ones((n,m))*(-99999999)
    preall[user,item]=pre
    id=np.argsort(preall,axis=1,kind='quicksort',order=None)
    id=id[:,::-1]
    id1=id[:,:atk]
    # print(id1)
    ans=ex.gaotest(posuser,positem,id1,id)
    # ans=mycalc(posuser,positem,id1,id,mode,bias)
    # pre@k, re@k, NDCG, MRR, NDCG@k
    # print(ans)
    # print('have_pos_user / all_user:', ans[5], len(list(set(user))))
    return ans[0],ans[1],ans[4]