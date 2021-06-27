# myapp.py
# top level
import pandas as pd
import umap
import pickle
from functools import partial
from math import floor
# ml
from sklearn import tree
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import EnsembleVoteClassifier
# bokeh
from bokeh.plotting import figure, curdoc  # , output_file, show
from bokeh.layouts import row
from bokeh.models import CheckboxButtonGroup
from bokeh.layouts import column, layout
from bokeh.models import Button, Div, Spinner, TextInput


# Class
class MyEnsemble:
    eclf = 0
    success = 0
    total = 0
    names = []
    weights = []
    name = ""


# Dataset

def load_csv(url):
    dataset = pd.read_csv(url)
    dataset = dataset.dropna()
    return dataset

reducer = umap.UMAP()

digits = load_digits()

Y = digits.target
X = digits.data

scaled_data = StandardScaler().fit_transform(X)

# Datos entrenamiento
X = scaled_data

# Separar en clases
j = 0

classes = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
color_maps = {
    0: "red", 1: "green", 2: "blue", 3: "brown", 4: "gray",
    5: "cyan", 6: "indigo", 7: "LightBlue", 8: "PaleGreen", 9: "Peru"
}

for i in range(len(Y)):
    classes[Y[i]].append(X[i])

X_train = []
Y_train = []
X_test = []
Y_test = []
limit = 0.7
for key, value in classes.items():
    a = len(value) * limit
    sub_X_train, sub_X_test = value[:floor(a)], value[floor(a):]
    sub_Y_train, sub_Y_test = [key] * floor(a), [key] * (len(value) - floor(a))
    X_train += sub_X_train
    X_test += sub_X_test
    Y_train += sub_Y_train
    Y_test += sub_Y_test

# Decision tree visualization
decision_tree = tree.DecisionTreeClassifier()
decision_tree = decision_tree.fit(X_train, Y_train)

# Random forest
random_forest = RandomForestClassifier(max_depth=2, random_state=0)
random_forest.fit(X_train, Y_train)

# Logistic Regression
logistic = LogisticRegression(random_state=0)
logistic = logistic.fit(X_train, Y_train)

# calcular precision

models = [decision_tree, random_forest, logistic]
predicted_results = [[], [], []]
success_list = []
total = len(Y_test)
model_name = 0
for model in models:
    success = 0
    for i in range(len(Y_test)):
        result = model.predict([X_test[i]])
        predicted_results[model_name].append(int(round(result[0])))
        if result == Y_test[i]:
            success += 1
    model_name += 1
    success_list.append(success)
    print(success, "correctos de ", total)

print("resultados", total, success_list)

print("Voting classifier")

# Visualization

# MODELS
reducer = umap.UMAP()
embedding = reducer.fit_transform(X_test)
colors2 = Y_test
color_maps = {
    0: "red", 1: "green", 2: "blue", 3: "brown", 4: "gray",
    5: "cyan", 6: "indigo", 7: "LightBlue", 8: "PaleGreen", 9: "Peru"
}

plot1 = figure(plot_width=200, plot_height=200)
plot1.title.text = "Decision Tree"

for i in range(len(embedding)):
    plot1.circle(embedding[i, 0], embedding[i, 1], size=8, color=color_maps[colors2[i]], alpha=1)
    plot1.circle(embedding[i, 0], embedding[i, 1], size=4, color=color_maps[predicted_results[0][i]], alpha=1)

print("plot 1 done")

plot2 = figure(plot_width=200, plot_height=200)
plot2.title.text = "Random Forest"

for i in range(len(embedding)):
    plot2.circle(embedding[i, 0], embedding[i, 1], size=8, color=color_maps[colors2[i]], alpha=1)
    plot2.circle(embedding[i, 0], embedding[i, 1], size=4, color=color_maps[predicted_results[1][i]], alpha=1)

print("plot 2 done")

plot3 = figure(plot_width=200, plot_height=200)
plot3.title.text = "Logistic Regression"
for i in range(len(embedding)):
    plot3.circle(embedding[i, 0], embedding[i, 1], size=8, color=color_maps[colors2[i]], alpha=1)
    plot3.circle(embedding[i, 0], embedding[i, 1], size=4, color=color_maps[predicted_results[2][i]], alpha=1)

print("plot 3 done")
plot_ensemble = figure(plot_width=200, plot_height=200)
plot_ensemble.circle(x=[], y=[], size=8, color="red", alpha=1)
plot_ensemble.circle(x=[], y=[], size=8, color="red", alpha=1)
plot_ensemble.title.text = "Ensamble"


# HTML

title_models = Div(
    text="""
        <div>
        <h2>Modelos</h2>
        <div>
        """,
    align="center"
)

title_ensemble = Div(
    text="""
        <h2>Ensamble</h2>
        """,
    align="center"
)

results_ensemble = Div(
    text="""
        <p>Resultados: </p>
        """,
    align="center"
)

title_control = Div(
    text="""
        <h2>Opciones</h2>
        """,
    align="center"
)

ensemble_list_div = Div(
    text="""
    <p> hola </p>
    """,
    align="center"
)


def gen_div(success, total):
    html_text = """<p>""" + str(success) + " correctos de " + str(total) + """</p>"""
    results_model = Div(
        text=html_text,
        align="center"
    )
    return results_model


def get_ensemble_list():
    index = open('ensambles/index.txt', 'r')
    lines = index.readlines()

    content = [x.strip() for x in lines]
    return content


def update_ensemble_list():
    ensemble_list = get_ensemble_list()
    text = """
        <table style="width:100%">
        <tr>
            <th>Nombre</th>
            <th>Modelos</th>
            <th>Pesos</th>
            <th>Correctos</th>
            <th>Total</th>
        </tr>
    """
    for ensemble in ensemble_list:
        text += "<tr><td>" + ensemble.split()[0] + "</td>"
        text += "<td>" + ensemble.split()[1] + "</td>"
        text += "<td>" + ensemble.split()[2] + "</td>"
        text += "<td>" + ensemble.split()[3] + "</td>"
        text += "<td>" + ensemble.split()[4] + "</td></th>"
    text += "</table>"

    return text

# Global ensemble vars

loading = False
current_ensemble = MyEnsemble()
comparison_ensemble = MyEnsemble()
comparison_ensembles_row = row(children=[], align="center")
ensemble_list_div.text = update_ensemble_list()

# Widgets


LABELS = ["decision tree", "random forest", "logistic"]

checkbox_button_group = CheckboxButtonGroup(labels=LABELS, active=[])
spinner_dt = Spinner(title="D. Tree", low=1, high=40, step=0.5, value=1, width=80, visible=False)
spinner_rf = Spinner(title="R. Forest", low=1, high=40, step=0.5, value=1, width=80, visible=False)
spinner_lr = Spinner(title="L. Regression", low=1, high=40, step=0.5, value=1, width=80, visible=False)
btn_weights = Button(label="Generar", button_type="success", visible=False)
btn_save = Button(label="Guardar ensamble", button_type="success", disabled=True, width=100, align="center")
btn_load = Button(label="Cargar ensamble", button_type="success", width=100, align="center")
btn_comparison = Button(label="Cargar ensamble", button_type="success", width=100, align="center")
name_input = TextInput(value="untitled", title="Nombre:", disabled=True, width=220, align="center")
load_input = TextInput(value="untitled", title="Nombre:", width=220, align="center")
comparison_input = TextInput(value="untitled", title="Nombre:", width=220, align="center")


# Callbacks

def call_back(attr, old, new):
    global loading
    if loading:
        loading = False
        return None

    current_ensemble.names = []
    if not new:
        btn_weights.visible = False
    else:
        btn_weights.visible = True
    if 0 in new:
        current_ensemble.names.append("dt")
        spinner_dt.visible = True
    else:
        spinner_dt.visible = False
    if 1 in new:
        current_ensemble.names.append("rf")
        spinner_rf.visible = True
    else:
        spinner_rf.visible = False
    if 2 in new:
        current_ensemble.names.append("lr")
        spinner_lr.visible = True
    else:
        spinner_lr.visible = False


def make_ensemble():
    current_ensemble.weights = []
    models_list = []
    for i in current_ensemble.names:
        if i == "dt":
            current_ensemble.weights.append(spinner_dt.value)
            models_list.append(decision_tree)
        if i == "rf":
            current_ensemble.weights.append(spinner_rf.value)
            models_list.append(random_forest)
        if i == "lr":
            current_ensemble.weights.append(spinner_lr.value)
            models_list.append(logistic)

    print("models list", models_list)
    print("weights", current_ensemble.weights)

    if current_ensemble.names:
        print(models_list, current_ensemble.weights)
        eclf = EnsembleVoteClassifier(clfs=models_list, weights=current_ensemble.weights, fit_base_estimators=False, voting='soft')

        eclf.fit(X_train, Y_train)  # No hace nada, por fit_base_estimators = false
        current_ensemble.eclf = eclf
        voting_results = eclf.predict(X_test)

        total = 0
        success = 0
        for i in range(len(Y_test)):
            if voting_results[i] == Y_test[i]:
                success += 1
            total += 1
        print(success, total)
        current_ensemble.success = success
        current_ensemble.total = total
        # Update ensamble plot
        # clear
        for i in range(0):  # esto demora mucho
            plot_ensemble.circle(embedding[i, 0], embedding[i, 1], size=8, color=color_maps[colors2[i]], alpha=1)
            plot_ensemble.circle(embedding[i, 0], embedding[i, 1], size=4, color=color_maps[voting_results[i]], alpha=1)

        print("resultados:", success, total)
        new_text = "<p> " + str(success) + " correctos de " + str(total) + "</p>"
        # print(new_text)
        results_ensemble.text = new_text
        name_input.disabled = False
        btn_save.disabled = False


def save_ensemble():
    print("current_ensemble")
    print(current_ensemble)
    print(current_ensemble.eclf)
    print(current_ensemble.names)
    print(current_ensemble.weights)
    print(current_ensemble.success, current_ensemble.total)

    # save to file
    current_ensemble.name = name_input.value
    with open('ensambles/' + name_input.value + '.pkl', 'wb') as output:
        pickle.dump(current_ensemble, output, pickle.HIGHEST_PROTOCOL)

    # update index
    with open("ensambles/index.txt", "a") as myfile:
        myfile.write(
            name_input.value + " " +
            ','.join(current_ensemble.names) + " " +
            ','.join([str(x) for x in current_ensemble.weights]) + " " +
            str(current_ensemble.success) + " " +
            str(current_ensemble.total) +
            "\n"
        )

    ensemble_list_div.text = update_ensemble_list()


def get_ensemble(input_object):
    global loading
    destination = MyEnsemble()
    loading = True
    with open('ensambles/' + input_object.value + '.pkl', 'rb') as abc:
        print("tmr: ", abc)
        destination = pickle.load(abc)
    print("loaded")
    print(destination)
    print(destination.eclf)
    print(destination.names, len(destination.names))
    print(destination.weights)
    print(destination.name)
    print(destination.success)
    print(destination.total)
    return destination


def update_ensemble_view():
    destination = get_ensemble(load_input)

    new_check_group = []
    if "dt" not in destination.names:
        spinner_dt.visible = False
    else:
        spinner_dt.visible = True
        weight_index = destination.names.index("dt")
        spinner_dt.value = destination.weights[weight_index]
        new_check_group.append(0)

    if "rf" not in destination.names:
        spinner_rf.visible = False
    else:
        spinner_rf.visible = True
        weight_index = destination.names.index("rf")
        spinner_rf.value = destination.weights[weight_index]
        new_check_group.append(1)

    if "lr" not in destination.names:
        spinner_lr.visible = False
    else:
        spinner_lr.visible = True
        weight_index = destination.names.index("lr")
        spinner_lr.value = destination.weights[weight_index]
        new_check_group.append(2)

    checkbox_button_group.active = new_check_group
    new_text = "<p> " + str(destination.success) + " correctos de " + str(destination.total) + "</p>"
    results_ensemble.text = new_text

    global current_ensemble
    current_ensemble = destination
    print("current_ensemble")
    print(current_ensemble)
    print(current_ensemble.name)
    print(current_ensemble.names)
    print(current_ensemble.success, current_ensemble.total)
    print(current_ensemble.weights)
    return destination


def update_comparison_view():
    global comparison_ensemble
    comparison_ensemble = get_ensemble(comparison_input)
    print("loading ensemble for comparison")
    print(comparison_ensemble)
    print(comparison_ensemble.eclf)
    print(comparison_ensemble.name)
    print(comparison_ensemble.names)
    print(comparison_ensemble.weights)
    new_plot = figure(plot_width=200, plot_height=200)
    new_plot.title.text = comparison_ensemble.name

    comparison_output = Div(align="center")
    comparison_output.text = "<p> Modelos: " + ','.join(comparison_ensemble.names) + "<br>" + \
                             str(comparison_ensemble.success) + ", " + str(comparison_ensemble.total) + "</p>"

    comparison_ensembles_row.children.append(
        column(
            children=[
                new_plot,
                comparison_output
            ]
        )
    )


def get_ensemble_handler(destination):
    if destination == "current":
        update_ensemble_view()
    if destination == "comparison":
        update_comparison_view()

checkbox_button_group.on_change("active", call_back)
btn_weights.on_click(make_ensemble)
btn_save.on_click(save_ensemble)
btn_load.on_click(
    partial(get_ensemble_handler, destination="current")
)
btn_comparison.on_click(
    partial(get_ensemble_handler, destination="comparison")
)
# Main
curdoc().add_root(layout(children=[
    [
        # Generar ensambles
        column(
            row(
                column(
                    title_models,
                    row(
                        column(plot1, gen_div(success_list[0], total)),
                        column(plot2, gen_div(success_list[1], total)),
                    ),
                    row(
                        column(plot3, gen_div(success_list[2], total)),
                        column(plot3, gen_div(success_list[2], total)),
                    ),
                ),
                column(
                    title_ensemble,
                    Div(text="<p>Modelos a usar: </p>", align="center"),
                    checkbox_button_group,
                    Div(text="<p>Pesos: </p>", align="center"),
                    row(spinner_dt, spinner_rf, spinner_lr),
                    btn_weights,
                    column(plot_ensemble),
                    results_ensemble
                ),
                column(
                    title_control,
                    row(
                        column(
                            Div(text="<p>Ensamble actual: </p>", align="center"),
                            name_input,
                            btn_save
                        ),
                        column(
                            Div(text="<p>Cargar ensamble: </p>", align="center"),
                            load_input,
                            btn_load
                        )
                    ),
                    Div(text="<p>Lista ensambles: </p>", align="center"),
                    ensemble_list_div
                )
            ),
            # Cargar ensambles
            column(
                children=[
                    Div(text="<h2> Comparar ensambles </h2>", align="center"),
                    row(
                        children=[
                            comparison_input, btn_comparison
                        ], align="center"),
                    comparison_ensembles_row
                ], align="center"
            )
        )
    ],
], width_policy="fit"))
