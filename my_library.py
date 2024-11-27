def test_load():
  return 'loaded'
  
def compute_probs(neg,pos):
  p0 = neg/(neg+pos)
  p1 = pos/(neg+pos)
  return [p0,p1]
  
def cond_prob(full_table, the_evidence_column, the_evidence_column_value, the_target_column, the_target_column_value):
  assert the_evidence_column in full_table
  assert the_target_column in full_table
  assert the_evidence_column_value in up_get_column(full_table, the_evidence_column)
  assert the_target_column_value in up_get_column(full_table, the_target_column)
  #your function body below - copy and paste then align with parameter names
  t_subset = up_table_subset(full_table, target, 'equals', target_value)
  e_list = up_get_column(t_subset, evidence)
  p_b_a = sum([1 if v==evidence_value else 0 for v in e_list])/len(e_list)
  return p_b_a + .01  #Laplace smoothing factor

def cond_probs_product(full_table, evidence_row, target_column, target_column_value):
  assert target_column in full_table
  assert target_column_value in up_get_column(full_table, target_column)
  assert isinstance(evidence_row, list)
  assert len(evidence_row) == len(up_list_column_names(full_table)) - 1   # - 1 because subtracting off the target column from full_table
  evidence_columns = up_list_column_names(full_table)[:-1]
  evidence_values = evidence_row  
  evidence_complete = up_zip_lists(evidence_columns, evidence_values)
  #your function body below
  cond_prob_list= []
  for e,v in evidence_complete:
   a_list= cond_prob(full_table, e, v, target_column, target_column_value)
   p_b_a= a_list
   cond_prob_list+=[p_b_a]
  product= up_product(cond_prob_list)
  return product

def prior_prob(full_table, the_column, the_column_value):
  assert the_column in full_table
  assert the_column_value in up_get_column(full_table, the_column)
  #your function body below
  t_list = up_get_column(full_table, the_column)
  p_a = sum([1 if v==the_column_value else 0 for v in t_list])/len(t_list)
  return p_a

def naive_bayes(full_table, evidence_row, target_column):
  assert target_column in full_table
  assert isinstance(evidence_row, list)
  assert len(evidence_row) == len(up_list_column_names(full_table)) - 1   # - 1 because subtracting off the target
  #compute P(target=0|...) by using cond_probs_product, finally multiply by P(target=0) using prior_prob
  p0=cond_probs_product(full_table, evidence_row, target_column, 0)* prior_prob(full_table, target_column, 0)
  #do same for P(target=1|...)
  p1=cond_probs_product(full_table, evidence_row, target_column, 1)* prior_prob(full_table, target_column, 1)
  #Use compute_probs to get 2 probabilities
  neg, pos = compute_probs(p0,p1)
  #return your 2 results in a list
  return [neg,pos]

def metrics(zipped_list):
  assert isinstance(zipped_list, list)
  assert all([isinstance(v, list) for v in zipped_list])
  assert all([len(v)==2 for v in zipped_list])
  assert all([isinstance(a,(int,float)) and isinstance(b,(int,float)) for a,b in zipped_list]), f'zipped_list contains a non-int or non-float'
  assert all([float(a) in [0.0,1.0] and float(b) in [0.0,1.0] for a,b in zipped_list]), f'zipped_list contains a non-binary value'

  #first compute the sum of all 4 cases. See code above
  tn = sum([1 if pair==[0,0] else 0 for pair in zipped_list])
  tp = sum([1 if pair==[1,1] else 0 for pair in zipped_list])
  fp = sum([1 if pair==[1,0] else 0 for pair in zipped_list])
  fn = sum([1 if pair==[0,1] else 0 for pair in zipped_list])

  #now can compute precicision, recall, f1, accuracy. Watch for divide by 0.
  precision = (tp/(tp+fp) if (tp+fp) else 0)
  recall = (tp/(tp+fn) if (tp+fn) else 0)
  f1 = (2*((precision*recall)/(precision+recall)) if (precision+recall) else 0)
  accuracy = ((tp+tn)/(tp+tn+fp+fn) if (tp+tn+fp+fn) else 0)

  #now build dictionary with the 4 measures - round values to 2 places
  predict= {'Precision': round(precision,2), 'Recall': round(recall,2), 'F1': round(f1,2), 'Accuracy': round(accuracy,2)}

  #finally, return the dictionary
  return predict

from sklearn.ensemble import RandomForestClassifier

def run_random_forest(train, test, target, n):
  #target is target column name
  #n is number of trees to use

  assert target in train   #have not dropped it yet
  assert target in test

  #your code below - copy, paste and align from above
  clf = RandomForestClassifier(n_estimators=n, max_depth=2, random_state=0)
  X = up_drop_column(scaled_train, 'adopted')
  y = up_get_column(scaled_train,'adopted')
  assert isinstance(y,list)
  assert len(y)==len(X)

  clf.fit(X, y)
  all_mets = []
  for t in thresholds:
    predictions = [1 if pos>t else 0 for pos in pos_probs]
    pred_act_list = up_zip_lists(predictions, k_actuals)
    mets = metrics(pred_act_list)
    mets['Threshold'] = t
    all_mets = all_mets + [mets]

  metrics_table = up_metrics_table(all_mets)

  return metrics_table

#I'll give you a start

def try_archs(train_table, test_table, target_column_name, architectures, thresholds):
  arch_acc_dict = {}  #ignore if not attempting extra credit

  #now loop through architectures
  for arch in architectures:
    probs = up_neural_net(train_table, test_table, arch, target_column_name)
    pos_probs = [pos for neg,pos in probs]
    all_mets = []
    #loop through thresholds
    for t in thresholds: 
      predictions = [1 if pos>=t else 0 for pos in pos_probs]
      scaled_test = test_table  
      pred_act_list = up_zip_lists(predictions, up_get_column(scaled_test, target_column_name))  
      mets = metrics(pred_act_list)
      mets['Threshold'] = t
      all_mets = all_mets + [mets]

    #arch_acc_dict[tuple(arch)] = max(...)  #extra credit - uncomment if want to attempt
    arch_acc_dict[tuple(arch)] = max([mets['Accuracy'] for mets in all_mets])
    print(f'Architecture: {arch}')
    display(up_metrics_table(all_mets))

  return arch_acc_dict
