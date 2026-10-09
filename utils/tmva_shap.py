"""TMVA Grad XML -> SHAP additive log-odds forest; validated against TMVA."""
import xml.etree.ElementTree as ET
import numpy as np


def forest_from_xml(path, columns):
    root=ET.parse(path).getroot()
    options={n.attrib['name']:n.text for n in root.find('Options')}
    if root.attrib['Method']!='BDT::BDTG' or options['BoostType']!='Grad' or options['VarTransform']!='None':
        raise ValueError('Only untransformed TMVA BDTG is supported')
    names=[n.attrib['Expression'] for n in root.find('Variables')]
    if names!=list(columns):raise ValueError('TMVA XML feature order mismatch')
    trees=[]
    for binary in root.find('Weights'):
        nodes=list(binary.find('Node').iter('Node'));lookup={id(n):i for i,n in enumerate(nodes)};size=len(nodes)
        left=np.full(size,-1,dtype=np.int32);right=left.copy();feature=left.copy()
        thresholds=np.zeros(size);values=np.zeros((size,1))
        for i,n in enumerate(nodes):
            if int(n.attrib['NCoef'])!=0:raise ValueError('Fisher splits unsupported')
            children=list(n)
            if children:
                sides={x.attrib['pos']:lookup[id(x)] for x in children}
                a,b=sides['l'],sides['r']
                if n.attrib['cType']=='0':a,b=b,a
                left[i],right[i]=a,b;feature[i]=int(n.attrib['IVar'])
                # TMVA takes the >= branch; SHAP's right branch is >.
                cut=float(n.attrib['Cut']);rounded=np.float32(cut)
                # Interventional SHAP internally stores thresholds as float32.
                # Use the largest float32 strictly below TMVA's double cut;
                # this preserves >= exactly for all TMVA float32 inputs.
                thresholds[i]=float(rounded if float(rounded)<cut else np.nextafter(rounded,np.float32(-np.inf)))
            else:
                # TMVA raw = tanh(sum(leaf response)); mapped score=logistic(2*sum).
                values[i,0]=2*float(n.attrib['res'])
        trees.append(dict(children_left=left,children_right=right,children_default=left.copy(),
            features=feature,thresholds=thresholds,values=values,node_sample_weight=np.ones(size)))
    return dict(trees=trees,base_offset=0.,objective='binary_crossentropy',tree_output='log_odds',input_dtype=np.float32,internal_dtype=np.float64)


def _explain_chunk(args):
    import shap
    forest,background,chunk=args
    explainer=shap.TreeExplainer(forest,data=background,
        feature_perturbation='interventional',model_output='raw')
    return explainer.shap_values(chunk,check_additivity=True)


def explain_bdtg(model, transformed, background_size=100):
    import shap
    forest=forest_from_xml(model.path,transformed.columns)
    background=transformed.sample(n=min(background_size,len(transformed)),random_state=42)
    explainer=shap.TreeExplainer(forest,data=background.to_numpy(dtype=np.float32),
        feature_perturbation='interventional',model_output='raw')
    values_input=transformed.to_numpy(dtype=np.float32)
    margin=explainer.model.predict(values_input)
    raw=model.predict_raw(transformed)
    np.testing.assert_allclose(np.tanh(margin/2),raw,atol=2e-12,rtol=0)
    if len(values_input)>1000:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=4) as pool:
            phi=np.concatenate(list(pool.map(_explain_chunk,
                [(forest,background.to_numpy(dtype=np.float32),chunk)
                 for chunk in np.array_split(values_input,4)])))
    else:
        phi=explainer.shap_values(values_input,check_additivity=True)
    residual=explainer.expected_value+phi.sum(axis=1)-margin
    np.testing.assert_allclose(explainer.expected_value+phi.sum(axis=1),margin,atol=2e-5,rtol=1e-6)
    metadata=dict(method='TreeSHAP interventional',explained_output='logit(mapped score)=2*sum(TMVA leaf responses)',
        background='uniform sample of the displayed selected population; seed42',background_indices=background.index.tolist(),
        background_size=len(background),expected_value=float(explainer.expected_value),
        tmva_response_max_error=float(np.max(np.abs(np.tanh(margin/2)-raw))),
        additivity_max_error=float(np.max(np.abs(residual))),
        note='TMVA XML has no training node covers; explicit empirical background is used. XGBoost path-dependent SHAP is unchanged.')
    return phi,metadata
