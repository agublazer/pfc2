# myapp.py
# top level
import pandas as pd
import umap
import pickle
# ml
from sklearn import tree
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
from bokeh.palettes import RdYlBu3


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

penguins = load_csv("https://github.com/allisonhorst/palmerpenguins/raw/5b5891f01b52ae26ad8cb9755ec93672f49328a8/data/penguins_size.csv")

reducer = umap.UMAP()

# Usar dataset de prueba con 4 atributos
penguin_data = penguins[
    [
        "culmen_length_mm",
        "culmen_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
    ]
].values

scaled_penguin_data = StandardScaler().fit_transform(penguin_data)
species = [x for x in penguins.species_short.map({"Adelie": 0, "Chinstrap": 1, "Gentoo": 2})]

# Datos entrenamiento
X = scaled_penguin_data
Y = species

class_1 = []
class_2 = []
class_3 = []

# Separar en clases
for i in range(len(species)):
    if species[i] == 0:
        class_1.append(scaled_penguin_data[i])
    if species[i] == 1:
        class_2.append(scaled_penguin_data[i])
    if species[i] == 2:
        class_3.append(scaled_penguin_data[i])

limit = 0.7
class_1_train = class_1[0:int(len(class_1) * limit)]
class_1_test = class_1[int(len(class_1) * limit):]

class_2_train = class_2[0:int(len(class_2) * limit)]
class_2_test = class_2[int(len(class_2) * limit):]

class_3_train = class_3[0:int(len(class_3) * limit)]
class_3_test = class_3[int(len(class_3) * limit):]

X_train = class_1_train + class_2_train + class_3_train
X_test = class_1_test + class_2_test + class_3_test

Y_train = [0] * len(class_1_train) + [1] * len(class_2_train) + [2] * len(class_3_train)
Y_test = [0] * len(class_1_test) + [1] * len(class_2_test) + [2] * len(class_3_test)

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
color_maps = {0: "red", 1: "green", 2: "blue"}

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
ensemble_list_div.text = update_ensemble_list()

# Widgets


LABELS = ["decision tree", "random forest", "logistic"]

checkbox_button_group = CheckboxButtonGroup(labels=LABELS, active=[])
spinner_dt = Spinner(title="D. Tree", low=1, high=40, step=0.5, value=1, width=80, visible=False)
spinner_rf = Spinner(title="R. Forest", low=1, high=40, step=0.5, value=1, width=80, visible=False)
spinner_lr = Spinner(title="L. Regression", low=1, high=40, step=0.5, value=1, width=80, visible=False)
btn_weights = Button(label="Generar", button_type="success", visible=False)
btn_save = Button(label="Guardar ensamble", button_type="success", disabled=True, width=80, align="center")
btn_load = Button(label="Cargar ensamble", button_type="success", width=80, align="center")
name_input = TextInput(value="untitled", title="Nombre:", disabled=True, width=160, align="center")
load_input = TextInput(value="untitled", title="Nombre:", width=160, align="center")


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

    """
    new_eclf = 123
    with open('ensambles/' + name_input.value + '.pkl', 'rb') as input:
        new_eclf = pickle.load(input)
    print("loaded", new_eclf)
    """


def load_ensemble():
    global current_ensemble
    global loading
    loading = True
    with open('ensambles/' + load_input.value + '.pkl', 'rb') as abc:
        print("tmr: ", abc)
        current_ensemble = pickle.load(abc)
    print("loaded")
    print(current_ensemble)
    print(current_ensemble.eclf)
    print(current_ensemble.names, len(current_ensemble.names))
    print(current_ensemble.weights)
    print(current_ensemble.name)
    print(current_ensemble.success)
    print(current_ensemble.total)

    new_check_group = []
    if "dt" not in current_ensemble.names:
        spinner_dt.visible = False
    else:
        spinner_dt.visible = True
        weight_index = current_ensemble.names.index("dt")
        spinner_dt.value = current_ensemble.weights[weight_index]
        new_check_group.append(0)

    if "rf" not in current_ensemble.names:
        spinner_rf.visible = False
    else:
        spinner_rf.visible = True
        weight_index = current_ensemble.names.index("rf")
        spinner_rf.value = current_ensemble.weights[weight_index]
        new_check_group.append(1)

    if "lr" not in current_ensemble.names:
        print("LR NOT IN NAMES")
        spinner_lr.visible = False
    else:
        print("LR IN NAMES")
        spinner_lr.visible = True
        print("STEP 1")
        weight_index = current_ensemble.names.index("lr")
        print("STEP 2")
        spinner_lr.value = current_ensemble.weights[weight_index]
        print("STEP 3")
        new_check_group.append(2)
        print("STEP 4")
        print("DONE")

    print("LR ", spinner_lr)
    print("LR ", spinner_lr.visible)
    print("LR ", spinner_lr.value)
    checkbox_button_group.active = new_check_group
    print("CB", checkbox_button_group)
    print("CB", checkbox_button_group.active)

checkbox_button_group.on_change("active", call_back)
btn_weights.on_click(make_ensemble)
btn_save.on_click(save_ensemble)
btn_load.on_click(load_ensemble)

# Main
curdoc().add_root(layout([
    [
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
    ]
]))
