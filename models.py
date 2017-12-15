# Import important stuff
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
# Only needed for setup
import tensorflow as tf
from pandas import Series

from sklearn.model_selection import train_test_split
######################### NEURAL NETWORK MODEL ###############################

def neural_network_model():
######################### DATA PROCEDDING ###############################
    training_data = pd.read_csv('./perma/HR_comma_sep.csv')
    testing_data = pd.read_csv('./temp/tempCsv.csv')
    X = testing_data

    sale = training_data.sales.unique()
    salary = training_data.salary.unique()
    sale_int =  list(range(1,len(sale)+1))  #['sales' 'accounting' 'hr' 'technical' 'support' 'management' 'IT' 'product_mng' 'marketing' 'RandD']
    salary_int =  list(range(1,len(salary)+1))  # ['low' 'medium' 'high']

    X.sales.replace(sale, sale_int, inplace=True)
    X.salary.replace(salary, salary_int, inplace=True)
    print(testing_data.shape)
    test_data = X.iloc[:,1:10]
    print(test_data.head())
    test_data = test_data.as_matrix()
    np.save("./perma/processed_data.npy",test_data)

######################### TESTING ###############################
    data = np.load("./perma/processed_data.npy")
    data_mean = np.reshape(np.mean(data[:, 0: 9], axis=0), (1, 9))
    data_std = np.reshape(np.std(data[:, 0: 9], axis=0), (1, 9))
    print(data_mean)


    # data_num = 12000
    data_test = data[:, 0:9]
    data_test = (data_test - data_mean) / data_std
    # label_test = data[data_num:, 9]

    with tf.Session() as sess:
        data_ph = tf.placeholder(tf.float32, [None, 9])
        # label_ph = tf.placeholder(tf.int32, [None])
        # label = tf.one_hot(label_ph, 2)
        fc1 = tf.contrib.layers.fully_connected(data_ph, 100, activation_fn=tf.nn.relu,
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(0.0001))
        fc2 = tf.contrib.layers.fully_connected(fc1, 100, activation_fn=tf.nn.relu,
                                                weights_regularizer=tf.contrib.layers.l2_regularizer(0.0001))
        logits = tf.contrib.layers.fully_connected(fc2, 2, activation_fn=None,
                                                   weights_regularizer=tf.contrib.layers.l2_regularizer(0.0001))
        predictions = tf.nn.softmax(logits)
        predictions = tf.argmax(predictions, 1)
        # accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predictions, 1), tf.argmax(label, 1)), tf.float32))

        tf.global_variables_initializer().run()

        dict = np.load('./perma/step_50000_0102AM_November_19_2017.npy').item()
        for var in tf.global_variables():
            dict_name = var.name
            if dict_name in dict:
                print('loaded var: %s' % dict_name)
                sess.run(var.assign(dict[dict_name]))
            else:
                print('missing var %s, skipped' % dict_name)

        # print(sess.run(accuracy, feed_dict={data_ph: data_test})#, label_ph: label_test}))
        # print(sess.run(predictions, feed_dict={data_ph: data_test}))
        data_pre = sess.run(predictions, feed_dict={data_ph: data_test})#, label_ph: label_test})
        print(data_pre)
        data_pre = np.reshape(data_pre, (testing_data.shape[0],1))

        # print(data[data_num:,0:12].shape)
        # data = np.concatenate((data[data_num:,0:10], data_pre) ,axis = 1)
        testing_data['left'] = Series(np.random.randn(testing_data.shape[0]), index=testing_data.index)
        # testing_data.assign(left = data_pre)
        testing_data.left = data_pre
        testing_data.sales.replace(sale_int, sale, inplace=True)
        print(testing_data.head())

        testing_data.to_csv('./temp/dataWithPreds.csv')

        # print(data.shape)
        # np.save("./temp/nn_predict.npy", data)

#######################################################################################


########################## RETURN REQUESTED DATA #############################
# Function for deciding the reasons the employee is likely leaving (these are
# interpreted from a decision tree shown in the porject report)
def reason_decider(department_data):

    employee_list = []
    for index, row in department_data.iterrows():
        if (row['satisfaction_level']>0.465 and row['time_spend_company']>4.5 and row['last_evaluation']>0.805 and
                row['average_montly_hours']>215.5):
            reason = "This employee has a high number of monthly hours"

        elif row['satisfaction_level']<=0.115 and row['number_project']>2.5:
            reason = "This worker has very low satisfaction with their job"

        elif (row['satisfaction_level']<=0.465 and row['number_project']<=2.5 and row['last_evaluation']<0.44):
            reason = "This worker has a somewhat low satisfaction and scored poorly on their last evaluation"

        else:
            reason = "It is not clear exactly why this worker may leave, further investigation is needed"

        if row['left']==1:
            reason_string = row['EmployeeID']+": Leave: "+reason

        else:
            reason_string = row['EmployeeID']+": Stay"

        employee_list.append(reason_string)

    return employee_list

# Function used below, it takes in a variable name and bins the data then spits
# out frequencies
def variable_data(variable, all_data):

    # Create bins for the variables, for number of projects this needs to be equal to the total
    # to avoid odd bin ranges and empty bins
    if variable == "number_project":
        bins = np.linspace(all_data[variable].min(), all_data[variable].max(), all_data[variable].max())

    # For all other variables, create 10 bins using the max and min as start and end points
    else:
        bins = np.linspace(all_data[variable].min(), all_data[variable].max(), 10)

    # Create group names for each of the bins (round to avoid overly long names)
    group_names = []
    for index in range(0, len(bins) - 1):
        group_names.append(str(round(bins[index],2)) + " to " + str(round(bins[index + 1], 2)))

    # Now populate the bins with the appropriate data
    bin_freq = []
    for bin_index in range(0, len(bins) - 1):

        count = 0
        for df_index, row in all_data.iterrows():

            if row[variable] >= bins[bin_index] and row[variable] < bins[bin_index + 1]:
                count = count + 1
            else:
                pass

        bin_freq.append(count)

    # Preparation of output, the javascript in the chart.js graphs in the web app
    # require javascript objects as data. This can be passed on by Python but
    # requires a very specific string pattern: '{"y":number, "label":"string}' for
    # each object (one per bar on the canvas.js bar chart).
    data_list = []
    for name, freq in zip(group_names, bin_freq):
        # Have to construct a "string javascript object" so that it can be JSON parsed
        # by the web app, giving a real javascript object which can populate a graph
        made_string = '{ \"y\": ' + str(freq) + ', \"label\": \"' + str(name) + '\"}'
        # append each of these (one for each bin) to a list
        data_list.append(made_string)

    # return the of "javascript object strings" as the output
    return data_list


# This function returns data depending on the department selected in the
# dashboard page for the web app. It returns data in a format that is
# readable by the graph javascript used in the web app pages.
def get_requested_data(department):

    all_data = pd.read_csv('./temp/dataWithPreds.csv')
    # all_data2 = np.load('./temp/nn_predict.np')
    # training_data = pd.read_csv('./temp/HR_comma_seq.csv')
    # sale = training_data.sales.unique()
    # Remake the dataframe with only the rows from the requested
    # department

    department_list = (all_data.sales.unique()).tolist()

    if department == 'all':
        department_data = all_data

    else:
        department_data = all_data.loc[all_data['sales'] == department]

    # Get data for all the different variables
    satis_data = variable_data('satisfaction_level', department_data)
    project_num_data = variable_data('number_project', department_data)
    eval_data = variable_data('last_evaluation', department_data)
    month_hours_data = variable_data('average_montly_hours', department_data)
    time_company_data = variable_data('time_spend_company', department_data)

    # Get the reasons for employees who are likely to leave
    leave_reasons = reason_decider(department_data)

    # Output of the function is the data ready to be exported to the web page
    return department_list, leave_reasons, satis_data, project_num_data, eval_data, month_hours_data, time_company_data

