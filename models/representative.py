"""Model adapters and training-fold preprocessing used in the benchmark."""
from pathlib import Path
import copy, random, sys, time, warnings
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'vendor'))
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, SplineTransformer

DEEP={'mlp','resnet','ft_transformer','tabm'}


def study_mae(y,p,groups):
    return float(pd.DataFrame({'g':np.asarray(groups),'e':np.abs(np.asarray(y)-np.asarray(p))}).groupby('g').e.mean().mean())


def classic(name,params,shared,seed,nums,cats):
    from sklearn.linear_model import Ridge,ElasticNet
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern,WhiteKernel
    from sklearn.svm import SVR
    p=dict(params);kw=dict(shared)
    steps=[('fill',SimpleImputer(strategy='median',keep_empty_features=True))]
    if name=='additive_spline':
        steps.append(('basis',SplineTransformer(n_knots=p.pop('n_knots'),**kw)))
    if name in {'ridge','elastic_net','knn','svr_rbf','gp_matern','additive_spline'}:
        steps.append(('scale',StandardScaler()))
    if name in {'ridge','additive_spline'}:model=Ridge(**p)
    elif name=='elastic_net':model=ElasticNet(**p,max_iter=20000,tol=1e-5,selection='cyclic')
    elif name=='knn':model=KNeighborsRegressor(**p,n_jobs=2)
    elif name=='decision_tree':model=DecisionTreeRegressor(**p,random_state=seed)
    elif name=='random_forest':model=RandomForestRegressor(**p,**kw,random_state=seed)
    elif name=='extratrees':model=ExtraTreesRegressor(**p,**kw,random_state=seed)
    elif name=='xgboost':
        from xgboost import XGBRegressor
        model=XGBRegressor(**p,**kw,random_state=seed,device='cpu')
    elif name=='lightgbm':
        from lightgbm import LGBMRegressor
        model=LGBMRegressor(**p,**kw,random_state=seed,verbosity=-1,deterministic=True,force_col_wise=True)
    elif name=='catboost':
        from catboost import CatBoostRegressor
        model=CatBoostRegressor(**p,**kw,random_seed=seed,loss_function='RMSE',verbose=False)
    elif name=='svr_rbf':model=SVR(**p,**kw,cache_size=128)
    elif name=='gp_matern':
        kernel=Matern(length_scale=p['length_scale'],length_scale_bounds='fixed',nu=kw['nu'])+WhiteKernel(p['noise_level'],noise_level_bounds='fixed')
        model=GaussianProcessRegressor(kernel=kernel,optimizer=None,normalize_y=True,alpha=1e-6,random_state=seed)
    else:raise ValueError(name)
    pre=ColumnTransformer([('num',Pipeline(steps),nums),('cat',Pipeline([
        ('fill',SimpleImputer(strategy='constant',fill_value='__MISSING__',keep_empty_features=True)),
        ('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cats)])
    return Pipeline([('preprocess',pre),('model',model)])


class NeuralInputs:
    def __init__(self,nums,cats):
        self.nums,self.cats=nums,cats
        self.num=Pipeline([('fill',SimpleImputer(strategy='median',keep_empty_features=True)),('scale',StandardScaler())])

    def fit(self,x):
        self.num.fit(x[self.nums])
        self.maps=[{v:i+1 for i,v in enumerate(sorted(x[c].fillna('__MISSING__').astype(str).unique()))} for c in self.cats]
        self.card=[len(m)+1 for m in self.maps]
        return self

    def transform(self,x):
        import torch
        num=torch.tensor(self.num.transform(x[self.nums]),dtype=torch.float32)
        cat=torch.tensor(np.column_stack([x[c].fillna('__MISSING__').astype(str).map(m).fillna(0).to_numpy(dtype=np.int64) for c,m in zip(self.cats,self.maps)]),dtype=torch.long) if self.cats else None
        return num,cat


def deep_fit_predict(name,p,shared,seed,x,y,xt,nums,cats,validation=None,epochs=None):
    import torch
    import torch.nn.functional as F
    from rtdl_revisiting_models import MLP,ResNet,FTTransformer
    from tabm import TabM
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    pre=NeuralInputs(nums,cats).fit(x)
    xn,xc=pre.transform(x);tn,tc=pre.transform(xt)
    width=p['width'];drop=p['dropout'];nblocks=shared['n_blocks']
    if name=='mlp':model=MLP(d_in=len(nums)+sum(pre.card),d_out=1,n_blocks=nblocks,d_block=width,dropout=drop)
    elif name=='resnet':model=ResNet(d_in=len(nums)+sum(pre.card),d_out=1,n_blocks=nblocks,d_block=width,d_hidden=None,d_hidden_multiplier=2.,dropout1=drop,dropout2=0.)
    elif name=='ft_transformer':model=FTTransformer(n_cont_features=len(nums),cat_cardinalities=pre.card,d_out=1,n_blocks=nblocks,d_block=width,attention_n_heads=8,attention_dropout=drop,ffn_d_hidden=None,ffn_d_hidden_multiplier=4/3,ffn_dropout=drop,residual_dropout=0.)
    elif name=='tabm':model=TabM.make(n_num_features=len(nums),cat_cardinalities=pre.card,d_out=1,n_blocks=nblocks,d_block=width,dropout=drop,k=32,arch_type='tabm')
    else:raise ValueError(name)

    def forward(a,b):
        if name in {'mlp','resnet'}:
            ohe=[F.one_hot(b[:,i],n).to(a.dtype) for i,n in enumerate(pre.card)] if b is not None else []
            return model(torch.cat([a]+ohe,dim=1)).reshape(-1)
        z=model(a,b)
        return z.squeeze(-1)

    def predict(a,b):
        model.eval();v=[]
        with torch.no_grad():
            for i in range(0,len(a),256):
                z=forward(a[i:i+256],b[i:i+256] if b is not None else None)
                if name=='tabm':z=z.mean(dim=1)
                v.append(z.numpy())
        return np.concatenate(v)*scale+center

    y=np.asarray(y,float);center=float(y.mean());scale=max(float(y.std()),1e-8)
    ys=torch.tensor((y-center)/scale,dtype=torch.float32)
    parameters=model.make_parameter_groups() if name=='ft_transformer' else model.parameters()
    optimizer=torch.optim.AdamW(parameters,lr=p['learning_rate'],weight_decay=shared['weight_decay'])
    maxepochs=int(epochs if epochs is not None else shared['max_epochs'])
    best,best_epoch,best_state,stale=np.inf,0,None,0
    trace=[]
    for epoch in range(1,maxepochs+1):
        model.train();perm=torch.randperm(len(xn))
        # Avoid singleton final minibatches for ResNet batch normalization.
        chunks=list(perm.split(shared['batch_size']))
        if len(chunks)>1 and len(chunks[-1])==1:chunks[-2]=torch.cat([chunks[-2],chunks[-1]]);chunks.pop()
        for idx in chunks:
            optimizer.zero_grad(set_to_none=True)
            z=forward(xn[idx],xc[idx] if xc is not None else None)
            target=ys[idx,None].expand_as(z) if name=='tabm' else ys[idx]
            loss=F.mse_loss(z,target)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
        if validation is not None:
            vy,vg=validation;vp=predict(tn,tc);score=study_mae(vy,vp,vg)
            if not np.isfinite(score):raise FloatingPointError('Nonfinite inner validation score')
            trace.append({'epoch':epoch,'study_mae':score})
            if score<best-1e-8:
                best,best_epoch,stale=score,epoch,0
                best_state=copy.deepcopy(model.state_dict())
            else:stale+=1
            if stale>=shared['patience']:break
    if validation is not None:
        model.load_state_dict(best_state)
    else:best_epoch=maxepochs
    pred=predict(tn,tc)
    return pred,dict(best_epoch=best_epoch,epochs_run=epoch,max_epochs=maxepochs,
                     budget_exhausted=bool(validation is not None and epoch==maxepochs),
                     validation_trace=trace,parameter_count=sum(q.numel() for q in model.parameters()),
                     numerical_inputs=len(nums),categorical_cardinalities=pre.card,
                     target_center=center,target_scale=scale)


def fit_predict(spec,params,seed,tr,xt,contract,validation=None,epochs=None):
    name=spec['model'];cols=contract['feature_columns'];ycol=contract['target']
    assert list(xt.columns)==cols and ycol not in xt
    assert set(cols).isdisjoint({'row_id','study_id','leakage_group_id',ycol})
    x=tr[cols].copy();xt=xt.copy()
    for col in contract['categorical_features']:
        x[col]=x[col].fillna('__MISSING__').astype(str);xt[col]=xt[col].fillna('__MISSING__').astype(str)
    start=time.monotonic()
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always')
        if name in DEEP:
            pred,details=deep_fit_predict(name,params,spec['shared_parameters'],seed,x,tr[ycol],xt,
                                         contract['numeric_features'],contract['categorical_features'],validation,epochs)
        else:
            m=classic(name,params,spec.get('shared_parameters',{}),seed,contract['numeric_features'],contract['categorical_features'])
            m.fit(x,tr[ycol]);pred=m.predict(xt);details={}
    pred=np.asarray(pred,float)
    assert pred.shape==(len(xt),) and np.isfinite(pred).all()
    details.update(seconds=time.monotonic()-start,warnings=sorted(set(str(w.message) for w in ws)))
    return pred,details
