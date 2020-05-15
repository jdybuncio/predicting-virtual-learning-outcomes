import pandas as pd
import numpy as np


#Load in Dataframes
df_courses = pd.read_csv('../data/courses.csv')
df_assessments = pd.read_csv('../data/assessments.csv')
df_vle = pd.read_csv('../data/vle.csv')
df_studentInfo = pd.read_csv('../data/studentInfo.csv')
df_studentRegistration = pd.read_csv('../data/studentRegistration.csv')
df_studentAssessment = pd.read_csv('../data/studentAssessment.csv')
df_studentVle_vle = pd.read_csv('../data/studentVle_by_vle.csv')
df_studentVle_day = pd.read_csv('../data/studentVle_by_day.csv')

# Functions to create DF for modeling
def add_courses(df_courses, df_studentInfo):

    df_studentInfo['module_presentation'] = df_studentInfo.code_module + '-' + df_studentInfo.code_presentation
    
    mask = (df_studentInfo.final_result == 'Pass') | (df_studentInfo.final_result == 'Distinction')
    df_studentInfo['final_result_edit'] = np.where(mask, 'Pass', 'Fail')

    courses = df_courses.add_prefix('courses_')
    result = pd.merge(df_studentInfo, courses, how='left', left_on=['code_module', 'code_presentation'], right_on=['courses_code_module', 'courses_code_presentation'])
    return result.drop(['courses_code_module', 'courses_code_presentation'], axis = 1)



def add_assessments(df_assessments, result):
    
    mask_tma = df_assessments['assessment_type'] == 'TMA'
    mask_cma = df_assessments['assessment_type'] == 'CMA'
    mask_exam = df_assessments['assessment_type'] == 'Exam'
    num_assessments = df_assessments.groupby(['code_module','code_presentation']).count()['id_assessment'].reset_index().rename(columns={'id_assessment':'num_assessments'})
    num_tma_assessments = df_assessments[mask_tma].groupby(['code_module','code_presentation']).count()['id_assessment'].reset_index().rename(columns={'id_assessment':'num_tma_assessments'}).fillna(0)
    num_cma_assessments = df_assessments[mask_cma].groupby(['code_module','code_presentation']).count()['id_assessment'].reset_index().rename(columns={'id_assessment':'num_cma_assessments'}).fillna(0)
    num_exam_assessments = df_assessments[mask_exam].groupby(['code_module','code_presentation']).count()['id_assessment'].reset_index().rename(columns={'id_assessment':'num_exam_assessments'}).fillna(0)
    
    result = pd.merge(result, num_assessments, how='left', on=['code_module', 'code_presentation'])
    result = pd.merge(result, num_tma_assessments, how='left', on=['code_module', 'code_presentation'])
    result = pd.merge(result, num_cma_assessments, how='left', on=['code_module', 'code_presentation'])
    result = pd.merge(result, num_exam_assessments, how='left', on=['code_module', 'code_presentation'])
    result = result.fillna(0)
    result.num_cma_assessments = result.num_cma_assessments.astype('int64')

    return result 

def add_vle(df_vle, result):
    vle = df_vle.groupby(['code_module','code_presentation','activity_type']).size().reset_index().rename(columns={0:'num_vle'})
    vle_totals = vle.pivot_table(values = 'num_vle', index = ['code_module','code_presentation'], columns = 'activity_type')
    vle_totals['total_num_vle'] = vle_totals.sum(axis=1)
    vle_totals = vle_totals.reset_index().fillna(0)
    vle_totals = vle_totals.add_prefix('vle_')

    result = pd.merge(result, vle_totals, how='left', left_on=['code_module', 'code_presentation'],right_on=['vle_code_module', 'vle_code_presentation'])
    result = result.drop(['vle_code_module', 'vle_code_presentation'], axis = 1)
    return pd.merge(result, df_studentRegistration, how='left', on=['code_module', 'code_presentation','id_student'])

def add_vle_click(df_studentVle_vle, df_vle, result):

    vle_actions = pd.merge(df_studentVle_vle, df_vle, how='left', on=['id_site', 'code_module','code_presentation'])

    vle_actions_early = vle_actions[(vle_actions.date<12) & (vle_actions.date>=0)]
    vle_actions_before = vle_actions[vle_actions.date<0]

    student_vle_agg_by_type = vle_actions.groupby(['id_student','code_presentation','code_module','activity_type']).sum()['sum_click']
    student_early_vle_agg_by_type = vle_actions_early.groupby(['id_student','code_presentation','code_module','activity_type']).sum()['sum_click']
    student_before_vle_agg_by_type = vle_actions_before.groupby(['id_student','code_presentation','code_module','activity_type']).sum()['sum_click']

    student_vle_agg_by_type = student_vle_agg_by_type.reset_index()
    student_early_vle_agg_by_type = student_early_vle_agg_by_type.reset_index()
    student_before_vle_agg_by_type = student_before_vle_agg_by_type.reset_index()

    student_vle_agg = student_vle_agg_by_type.pivot_table(values = 'sum_click', index = ['id_student','code_presentation','code_module'], columns = 'activity_type')
    student_early_vle_agg = student_early_vle_agg_by_type.pivot_table(values = 'sum_click', index = ['id_student','code_presentation','code_module'], columns = 'activity_type')
    student_before_vle_agg = student_before_vle_agg_by_type.pivot_table(values = 'sum_click', index = ['id_student','code_presentation','code_module'], columns = 'activity_type')

    student_vle_agg['total_num_vle_actions'] = student_vle_agg.sum(axis=1)
    student_early_vle_agg['total_num_vle_actions'] = student_early_vle_agg.sum(axis=1)
    student_before_vle_agg['total_num_vle_actions'] = student_before_vle_agg.sum(axis=1)

    student_vle_agg = student_vle_agg.reset_index()
    student_early_vle_agg = student_early_vle_agg.reset_index()
    student_before_vle_agg = student_before_vle_agg.reset_index()

    student_vle_agg = student_vle_agg.fillna(0)
    student_early_vle_agg = student_early_vle_agg.fillna(0)
    student_before_vle_agg = student_before_vle_agg.fillna(0)

    student_vle_agg = student_vle_agg.add_prefix('all_click_')
    student_early_vle_agg = student_early_vle_agg.add_prefix('early_click_')
    student_before_vle_agg = student_before_vle_agg.add_prefix('before_click_')

    result = pd.merge(result, student_vle_agg, how='left', left_on=['code_module', 'code_presentation','id_student'],right_on=['all_click_code_module', 'all_click_code_presentation','all_click_id_student'])
    result = pd.merge(result, student_early_vle_agg , how='left', left_on=['code_module', 'code_presentation','id_student'],right_on=['early_click_code_module', 'early_click_code_presentation','early_click_id_student'])
    result = pd.merge(result, student_before_vle_agg , how='left', left_on=['code_module', 'code_presentation','id_student'],right_on=['before_click_code_module', 'before_click_code_presentation','before_click_id_student'])

    return result.drop(['all_click_id_student','all_click_code_presentation', 'all_click_code_module','early_click_id_student','early_click_code_presentation', 'early_click_code_module', 'before_click_id_student','before_click_code_presentation', 'before_click_code_module'],axis =1)


def add_vle_days(df_studentVle_day, df_vle, result):
    vle_days = pd.merge(df_studentVle_day, df_vle, how='left', on=['id_site', 'code_module','code_presentation'])
    vle_days = vle_days.drop(['week_from','week_to'], axis = 1)
    daily_early_txn_vle = vle_days[(vle_days.date<12) & (vle_days.date>=0)]
    daily_before_txn_vle = vle_days[vle_days.date<0]

    daily_early_txn_by_type = daily_early_txn_vle.groupby(['id_student','code_presentation','code_module','activity_type']).size()
    daily_before_txn_by_type = daily_before_txn_vle.groupby(['id_student','code_presentation','code_module','activity_type']).size()

    daily_early_txn_by_type = daily_early_txn_by_type.reset_index()
    daily_before_txn_by_type =daily_before_txn_by_type.reset_index()

    student_early_txn_agg = daily_early_txn_by_type.pivot_table(values = 0, index = ['id_student','code_presentation','code_module'], columns = 'activity_type')
    student_before_txn_agg = daily_before_txn_by_type.pivot_table(values = 0, index = ['id_student','code_presentation','code_module'], columns = 'activity_type')

    student_early_txn_agg['total_num_vle_days'] = student_early_txn_agg.sum(axis=1)
    student_before_txn_agg['total_num_vle_days'] = student_before_txn_agg.sum(axis=1)


    student_early_txn_agg = student_early_txn_agg.reset_index()
    student_before_txn_agg = student_before_txn_agg.reset_index()

    student_early_txn_agg = student_early_txn_agg.fillna(0)
    student_before_txn_agg = student_before_txn_agg.fillna(0)


    student_early_txn_agg = student_early_txn_agg.add_prefix('early_days_')
    student_before_txn_agg = student_before_txn_agg.add_prefix('before_days_')


    result = pd.merge(result, student_early_txn_agg , how='left', left_on=['code_module', 'code_presentation','id_student'],right_on=['early_days_code_module', 'early_days_code_presentation','early_days_id_student'])
    result = pd.merge(result, student_before_txn_agg , how='left', left_on=['code_module', 'code_presentation','id_student'],right_on=['before_days_code_module', 'before_days_code_presentation','before_days_id_student'])
    return result.drop(['early_days_id_student','early_days_code_presentation', 'early_days_code_module', 'before_days_id_student','before_days_code_presentation', 'before_days_code_module'],axis =1)


def pre_processing(df_no_w):
    df_no_w = df_no_w.drop(['final_result', 'code_module', 'code_presentation', 'id_student'], axis = 1)

    df_no_w["highest_education"] = df_no_w["highest_education"].map({np.nan: 0, 'No Formal quals': 1, "Lower Than A Level": 2, \
                                                                "A Level or Equivalent": 3, 'HE Qualification': 4, 'Post Graduate Qualification': 5}).astype(int)

    df_no_w["imd_band"] = df_no_w["imd_band"].map({'0': 0, '0-10%': 1, '10-20': 2, \
                                                '20-30%': 3, '30-40%': 4, '40-50%': 5, \
                                                '50-60%': 6, '60-70%': 7, '70-80%': 8, \
                                                '80-90%': 9, '90-100%': 10 \
                                                })

    df_no_w["age_band"] = df_no_w["age_band"].map({'0-35': 0, '35-55': 1, '55<=': 2 \
                                                }).astype(int)
    df_no_w["disability"]= df_no_w["disability"].map({'Y': 1, 'N': 0 \
                                                }).astype(int)
    df_no_w["gender"]= df_no_w["gender"].map({'M': 1, 'F': 0 \
                                                }).astype(int)
    df_no_w["final_result_edit"]= df_no_w["final_result_edit"].map({'Fail': 1, 'Pass': 0 \
                                                }).astype(int)
    df_no_w.rename(columns = {'final_result_edit' : 'fail_flag'}, inplace = True)

    df_no_w = pd.get_dummies(df_no_w)

    return df_no_w



def df_day_twelve(df):
    click_cols = ['all_click_dataplus',
                    'all_click_dualpane',
                    'all_click_externalquiz',
                    'all_click_folder',
                    'all_click_forumng',
                    'all_click_glossary',
                    'all_click_homepage',
                    'all_click_htmlactivity',
                    'all_click_oucollaborate',
                    'all_click_oucontent',
                    'all_click_ouelluminate',
                    'all_click_ouwiki',
                    'all_click_page',
                    'all_click_questionnaire',
                    'all_click_quiz',
                    'all_click_repeatactivity',
                    'all_click_resource',
                    'all_click_sharedsubpage',
                    'all_click_subpage',
                    'all_click_url',
                    'all_click_total_num_vle_actions']

    df = df.drop(click_cols,axis = 1)

    # Get rid of any students who dropped out before Day 12
    df = df[(df.date_unregistration>=12) | (df.date_unregistration.isna())]

    df = df.drop(['date_unregistration'],axis = 1)
    return df





def df_day_zero(df):
    click_cols = ['all_click_dataplus',
                    'all_click_dualpane',
                    'all_click_externalquiz',
                    'all_click_folder',
                    'all_click_forumng',
                    'all_click_glossary',
                    'all_click_homepage',
                    'all_click_htmlactivity',
                    'all_click_oucollaborate',
                    'all_click_oucontent',
                    'all_click_ouelluminate',
                    'all_click_ouwiki',
                    'all_click_page',
                    'all_click_questionnaire',
                    'all_click_quiz',
                    'all_click_repeatactivity',
                    'all_click_resource',
                    'all_click_sharedsubpage',
                    'all_click_subpage',
                    'all_click_url',
                    'all_click_total_num_vle_actions']

    df = df.drop(click_cols,axis = 1)

    str_cols = ['early_days_dataplus',
                    'early_days_dualpane',
                    'early_days_externalquiz',
                    'early_days_forumng',
                    'early_days_glossary',
                    'early_days_homepage',
                    'early_days_htmlactivity',
                    'early_days_oucollaborate',
                    'early_days_oucontent',
                    'early_days_ouelluminate',
                    'early_days_ouwiki',
                    'early_days_page',
                    'early_days_questionnaire',
                    'early_days_quiz',
                    'early_days_resource',
                    'early_days_sharedsubpage',
                    'early_days_subpage',
                    'early_days_url',
                    'early_days_total_num_vle_days',
                    'early_click_dataplus',
                    'early_click_dualpane',
                    'early_click_externalquiz',
                    'early_click_forumng',
                    'early_click_glossary',
                    'early_click_homepage',
                    'early_click_htmlactivity',
                    'early_click_oucollaborate',
                    'early_click_oucontent',
                    'early_click_ouelluminate',
                    'early_click_ouwiki',
                    'early_click_page',
                    'early_click_questionnaire',
                    'early_click_quiz',
                    'early_click_resource',
                    'early_click_sharedsubpage',
                    'early_click_subpage',
                    'early_click_url',
                    'early_click_total_num_vle_actions']

    df = df.drop(str_cols,axis = 1)

    # Get rid of any students who dropped out before Day 0
    df = df[(df.date_unregistration>=0) | (df.date_unregistration.isna())]
    df = df.drop(['date_unregistration'],axis = 1)
    return df



if __name__ == "__main__":
    result = add_courses(df_courses,df_studentInfo)
    result = add_assessments(df_assessments, result)
    result = add_vle(df_vle, result)
    result = add_vle_click(df_studentVle_vle, df_vle, result)
    result = add_vle_days(df_studentVle_day, df_vle, result)

    df_pre_processed = pre_processing(result)
    result_day_twelve = df_day_twelve(df_pre_processed)
    result_day_zero = df_day_zero(df_pre_processed)

    result.to_csv('../data/joined_dataframe.csv')
    result_day_zero.to_csv('../data/dataset_for_modeling_day_zero.csv')
    result_day_twelve.to_csv('../data/dataset_for_modeling_day_twelve.csv')
