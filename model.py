from turtle import ycor
from unittest.mock import patch
from mesa.datacollection import DataCollector
from mesa import Model
from mesa.time import BaseScheduler
from mesa_geo.geoagent import GeoAgent, AgentCreator
from mesa_geo import GeoSpace
from shapely.geometry import Point
import math
import scipy.stats as stats
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
#from myPatch import MyPatch

########################## AGENTE HUMANOS ##########################
class PersonAgent(GeoAgent):
    """Person Agent."""
    cont = 0
    # Constructor
    def __init__(
        self,
        unique_id,
        model,
        geometry,
        crs,
        infectionState = None,
        age = None,
        timeSinceSuccesfullBite = None,
        timeSinceInfection = None,
        activities = None,
        homeLocation = None,
    ):
        super().__init__(unique_id, model, geometry, crs)
        # Initialization susceptibles humans (all human agents initialize as susceptible)
        infectionState = "susceptible"
        timeSinceSuccesfullBite = None
        timeSinceInfection = None
        age = self.ageSetter(random.random())
        activities = []
        homeLocation = [self.geometry.x, self.geometry.y]

        # Initialization infected humans (a portion of)
        if InfectedModel.agentCreated >= (self.model.totalHumans - self.model.infectedHumans):
            infectionState = "infected"
            timeSinceSuccesfullBite = 0
            timeSinceInfection = 0
            age = self.ageSetter(random.random())
            activities = []
            homeLocation = [self.geometry.x, self.geometry.y]

        self.infectionState = infectionState
        self.age = age
        self.timeSinceSuccesfullBite = timeSinceSuccesfullBite
        self.timeSinceInfection = timeSinceInfection
        self.activities = activities
        self.homeLocation = homeLocation
        self.is_human = True

        # Define constants
        self.naturalEmergenceRate = 0.3
        self.deathRate = 0.071428571428571
        self.mosquitoCarryingCapacity = 1000
        self.mosquitoBiteDemand = 0.5
        self.maxBitesPerHuman = 19
        self.probabilityOfTransmissionHToM = 0.333
        self.probabilityOfTransmissionMToH = 0.333
        self.distTiempoIncubacion = stats.lognorm(0.139, 1.099) # se crea una distribucion lognormal para generar el tiempo de incubacion
        self.distTiempoInfeccion = stats.uniform(3, 7) # se pone la duracion del tiempo de infeccion como una dist uniforme entre 3-7 dias

    def move_point(self, dx, dy):
        """
        Move a point by creating a new one
        :param dx:  Distance to move in x-axis
        :param dy:  Distance to move in y-axis
        """
        #print("============================================")
        #print("Estaba en x: ", self.geometry.x, " Estaba en y: ", self.geometry.y)
        #print("============================================")
        #print("Estoy en x: ", x, " Estoy en y: ", y)
        #print("============================================")
        return Point(dx , dy)

    # correr cada humano
    def step(self):
        """Advance one step."""
        #print('============================= EMPIEZA UN HUMANO =================================== ')
        tick = int(self.model.steps)
        if (tick % 3) == 1:
            activity = self.activities[1]
        elif (tick % 3) == 2:
            activity = self.activities[0]
        elif (tick % 3) == 0:
            activity = [self.homeLocation[0], self.homeLocation[1]]

        # human = self.model.space.get_intersecting_agents(self)
        # print("Arreglo De Humanos ")
        # patch_actual = [
        #     h for h in human if True #patch.temperature == 0
        # ]
        # print("patch actual: ", patch_actual)

        patch_actual_coord = Point(self.geometry.x, self.geometry.y)
        x = activity[0]
        y = activity[1]
        print('actividad x: ', x, 'actividad y: ', y)

        print("Actualmente estoy en ", self.geometry.x, " ", self.geometry.y)
        patches = self.model.space.get_intersecting_agents(self)
        patch_actual = [
            patch for patch in patches if True #patch.temperature == 0
        ]

        self.geometry = self.move_point(x,y)

        print("Nuevo arreglo(actual) ", self.unique_id)
        print(patch_actual)
        print("Patch donde estoy: ", patch_actual[0].unique_id, " Coordenadas: ", patch_actual_coord)
        self.actualizeSEIRStatus(patch_actual[0])
        if self.is_human == False:
            susceptible = patch_actual[0].m_susceptible
            infected = patch_actual[0].m_infected
            temp = patch_actual[0].temperature
            typeZone = patch_actual[0].typeZone
            print('tipo de zona', typeZone)
            #print('=====================TICK=========================== ', tick)

            if tick == 1:
                totalMosquitoes0 = patch_actual[0].totalMosquitoes[tick]
                totalMosquitoes1 = patch_actual[0].totalMosquitoes[tick]
            else:
                totalMosquitoes0 = patch_actual[0].totalMosquitoes[tick-1]
                totalMosquitoes1 = patch_actual[0].totalMosquitoes[tick]
            # Se crea un obj tipo patch
            patch = MyPatch(self.model, self.geometry, susceptible, infected, typeZone, totalMosquitoes0, totalMosquitoes1)
            resp = patch.step()
            patch_actual[0].m_susceptible = resp[0]
            patch_actual[0].m_infected = resp[1]
            patch_actual[0].total = resp[2]

        #self.geometry = self.move_point(self.homeLocation[0], self.homeLocation[1])
        self.actualizeTimes()

    def __repr__(self):
        return "Person " + str(self.unique_id)

    def actualizeTimes(self):
        if self.timeSinceSuccesfullBite != None:
            self.timeSinceSuccesfullBite += 1

        if self.timeSinceInfection != None:
            self.timeSinceInfection += 1

    def countHumansInPatch(self, patch):
        HumansinP = self.model.space.get_intersecting_agents(self)
        contH = 0
        for h in HumansinP:
            print('humans in patch ',patch, h)
            contH = contH + 1
        return contH-1

    def calculateInfectionProbabilityHuman(self, neigh_patch):
        #obtengo el numero de mosquitos de cada tipo del patch en el que estoy parada
        susceptibleMosquitoes = neigh_patch.m_susceptible
        infectedMosquitoes = neigh_patch.m_infected

        #obtengo el numero de humanos que hay en el patch que estoy parada
        totalHumans = self.countHumansInPatch(neigh_patch)
        print('total humans in patch:', neigh_patch, totalHumans)

        #contar cuantos mosquitos hay en el patch que estoy parada
        totalMosquitoes = susceptibleMosquitoes + infectedMosquitoes
        #print("Humanos en el patch ", totalHumans)
        successfulBitesPerHuman = 0
        if totalHumans > 0:
            totalSuccesfulBites = (self.mosquitoBiteDemand*totalMosquitoes*self.maxBitesPerHuman*totalHumans)/(self.mosquitoBiteDemand*totalMosquitoes+self.maxBitesPerHuman*totalHumans)
            #print("Demands bites: ", self.mosquitoBiteDemand, " total Mosquitoes ", totalMosquitoes, " max bites H ", self.maxBitesPerHuman)
            successfulBitesPerHuman = totalSuccesfulBites/totalHumans
        #print("Successfulbites ", successfulBitesPerHuman, " Probability ", self.probabilityOfTransmissionHToM, " Infected Mosquitoes ", infectedMosquitoes)
        infectionRateHumans = self.probabilityOfTransmissionMToH*successfulBitesPerHuman*infectedMosquitoes/totalMosquitoes
        #print("Infection Rate ", infectionRateHumans)
        humanInfectionProbability = 1-math.exp(-infectionRateHumans)
        #print('human infection prob', humanInfectionProbability)
        return humanInfectionProbability

    def actualizeSEIRStatus(self, patch):
        # si la persona esta en estado suceptible se calcula la probabilidad de pasar a infectado y se determina si pasa a infectado o no
        aleat = random.random()
        print("==================================")
        #print('NUMERO ALEAT', aleat)
        if self.infectionState == "susceptible":
            probabilityIOfInfectionHuman = self.calculateInfectionProbabilityHuman(patch)
            #print("Probabilidad vs aleat", aleat <= probabilityIOfInfectionHuman)
            print('PROBABILIDAD DE INFECCION', probabilityIOfInfectionHuman)
            if aleat <= probabilityIOfInfectionHuman:
                self.infectionState = "exposed"
                self.timeSinceSuccesfullBite = 0
                self.model.counts["exposed"] += 1
                self.model.counts["susceptible"] -= 1

        # si la persona esta en estado expuesto se determina si pasa a infectado
        elif self.infectionState == "exposed":
            acumProb = self.distTiempoIncubacion.cdf(self.timeSinceSuccesfullBite)
            #print("Time Bite ", self.timeSinceSuccesfullBite)
            #print("Probabilidad infectados vs aleat ", aleat <= acumProb)

            #print('ACUM PROB EXP-INF', acumProb)
            if aleat <= acumProb:
                self.infectionState = "infected"
                self.timeSinceInfection = 0
                self.model.counts["infected"] += 1
                self.model.counts["exposed"] -= 1

        # si la persona esta infectada se determina si pasa a recuperado
        elif self.infectionState == "infected":
            acumProb = self.distTiempoInfeccion.cdf(self.timeSinceInfection)
            #print('ACUM PROB INF-REC', acumProb)
            if aleat <= acumProb:
                print("Me cure")
                self.infectionState = "recovered"
                self.model.counts["infected"] -= 1
                self.model.counts["recovered"] += 1

    def ageSetter(self, prob):
        if 0.0811 <= prob and prob < 0.1615:
            return random.randint(5, 9)
        elif 0.1615 <= prob and prob < 0.2435:
            return random.randint(10, 14)
        elif 0.2435 <= prob and prob < 0.3292:
            return random.randint(15, 19)
        elif 0.3292 <= prob and prob < 0.4249:
            return random.randint(20, 24)
        elif 0.4249 <= prob and prob < 0.5196:
            return random.randint(25, 29)
        elif 0.5196 <= prob and prob < 0.6025:
            return random.randint(30, 34)
        elif 0.6025 <= prob and prob < 0.6801:
            return random.randint(35, 39)
        elif 0.6801 <= prob and prob < 0.7506:
            return random.randint(40, 44)
        elif 0.7506 <= prob and prob < 0.8097:
            return random.randint(45, 49)
        elif 0.8097 <= prob and prob < 0.8626:
            return random.randint(50, 54)
        elif 0.8626 <= prob and prob < 0.9062:
            return random.randint(55, 59)
        elif 0.9062 <= prob and prob < 0.9378:
            return random.randint(60, 64)
        elif 0.9378 <= prob and prob < 0.9601:
            return random.randint(65, 69)
        elif 0.9601 <= prob and prob < 0.9768:
            return random.randint(70, 74)
        elif 0.9768 <= prob and prob < 0.9890:
            return random.randint(75, 79)
        elif 0.9890 <= prob and prob <= 1:
            return random.randint(80, 90)
        else:
            return random.randint(0, 4)


########################## AGENTE PATCHES ##########################
class NeighbourhoodAgent(GeoAgent):
    """Neighbourhood agent. Changes color according to number of infected inside it."""

    def __init__(
        self,
        unique_id,
        model,
        geometry,
        crs,
        m_susceptible = None,
        m_infected = None,
        total = None,
        temperature = None,
        typeZone = None,
        totalMosquitoes = [],
    ):
        super().__init__(unique_id, model, geometry, crs)
        # Initialization patches
        m_susceptible = random.randint(1,100)
        m_infected = random.randint(0,5)
        total = m_susceptible + m_infected
        totalMosquitoes.append(total)
        temperature = self.define_temperatures(tick=0)
        num_rand_aux = random.randint(1,4)
        if num_rand_aux == 1:
            typeZone = "residential"
        elif num_rand_aux == 2:
            typeZone = "study"
        elif num_rand_aux == 3:
            typeZone = "work"
        elif num_rand_aux == 4:
            typeZone = "leisure"

        self.m_susceptible = m_susceptible
        self.m_infected = m_infected
        self.total = total
        self.temperature = temperature
        self.typeZone = typeZone
        self.totalMosquitoes = totalMosquitoes
        self.is_human = False

    def define_temperatures(self, tick):
        #file = pd.read_excel(r'.\temperatura y precipitacion bello.xlsx')
        max_temp = 21#file['Temperatura max'][tick]
        min_temp = 11#file['Temperatura min'][tick]
        temp = random.randint(min_temp, max_temp)
        return temp


    # actalizacion de los patches en cada tick
    def step(self):
        """Advance agent one step."""
        self.temperature = random.randint(25,30)

        # cojo el valor de mosquitos suceptibles, infectados y expuestos del dia anterior que utilizare para calcular el de este dia
        susc = self.m_susceptible
        inf = self.m_infected
        tick = int(self.model.steps)
        if tick == 1:
            mosq0 = self.totalMosquitoes[tick]
            mosq1 = self.totalMosquitoes[tick]
        else:
            mosq0 = self.totalMosquitoes[tick-1]
            mosq1 = self.totalMosquitoes[tick]
        type = self.typeZone

        patch = MyPatch(self.model, self.geometry, susc, inf, type, mosq0, mosq1)
        resp = patch.step()
        self.m_susceptible = resp[0]
        self.m_infected = resp[1]
        self.total = resp[2]
        self.totalMosquitoes.append(self.total)
        self.temperature = self.define_temperatures(tick)


    def __repr__(self):
        return "Neighborhood " + str(self.unique_id)


########################## SIMULATION BUILDER ##########################
class InfectedModel(Model):
    """Model class for a simplistic infection model."""

    # Geographical parameters for desired map
    MAP_COORDS = [6.333, -75.55]  # Bello
    geojson_regions = "bello_grid.geojson"
    unique_id = "id"
    agentCreated = 0

    def __init__(self, simulationTime, totalHumans, infectedHumans):

        self.schedule = BaseScheduler(self)
        self.space = GeoSpace()
        self.steps = 0
        self.counts = None
        self.reset_counts()

        # SIR model parameters
        self.simulationTime = simulationTime
        self.totalHumans = totalHumans
        self.infectedHumans = infectedHumans
        #self.counts["susceptible"] = totalHumans-infectedHumans
        #self.counts["infected"] = infectedHumans

        self.running = True
        self.datacollector = DataCollector(
            {
                "susceptible": get_susceptible_count,
                "exposed": get_exposed_count,
                "infected": get_infected_count,
                "recovered": get_recovered_count,
            }
        )

        # Set up the Neighbourhood patches for every region in file (add to schedule later)
        AC = AgentCreator(NeighbourhoodAgent, model= self)
        neighbourhood_agents = AC.from_file(self.geojson_regions, unique_id=self.unique_id)
        self.space.add_agents(neighbourhood_agents)
        # From all the neighbourhood agents, create a list with only the residential agents for initialization humans in home location
        neighbourhood_residential_agents = []
        neighbourhood_work_agents = []
        neighbourhood_study_agents = []
        neighbourhood_leisure_agents = []
        for i in range(len(neighbourhood_agents)):
            if neighbourhood_agents[i].typeZone == "residential":
                neighbourhood_residential_agents.append(neighbourhood_agents[i])
            elif neighbourhood_agents[i].typeZone == "work":
                neighbourhood_work_agents.append(neighbourhood_agents[i])
            elif neighbourhood_agents[i].typeZone == "study":
                neighbourhood_study_agents.append(neighbourhood_agents[i])
            elif neighbourhood_agents[i].typeZone == "leisure":
                neighbourhood_leisure_agents.append(neighbourhood_agents[i])

        # Generate PersonAgent population
        ac_population = AgentCreator(
            PersonAgent,
            model=self,
            crs = self.space.crs,
            )
        # Generate random location in the residential patches, add agent to space and scheduler
        cont_susc=0
        cont_inf=0
        print("Hola estoy en infected model.............")
        for i in range(totalHumans):
            # Se le asigna la home location a cada humano
            home_neighbourhood = self.random.randint(0, len(neighbourhood_residential_agents) - 1)  # Region where agent starts
            center_x, center_y = neighbourhood_residential_agents[home_neighbourhood].geometry.centroid.coords.xy
            this_bounds = neighbourhood_residential_agents[home_neighbourhood].geometry.bounds
            spread_x = int(this_bounds[2] - this_bounds[0])  # Heuristic for agent spread in region
            spread_y = int(this_bounds[3] - this_bounds[1])
            this_x = center_x[0] + self.random.randint(0, spread_x) - spread_x / 2
            this_y = center_y[0] + self.random.randint(0, spread_y) - spread_y / 2
            #this_person = ac_population.create_agent(Point(this_x, this_y), "P" + str(i))
            this_person = ac_population.create_agent(Point(-8416250.600374313, 711265.7371532868), "P" + str(i))
            InfectedModel.agentCreated = i

            # Se le asigna la actividad de estudio a humanos menores a 24 años
            if this_person.age <= 24:
                study_neighbourhood = self.random.randint(0, len(neighbourhood_study_agents) - 1)
                center_x, center_y = neighbourhood_study_agents[study_neighbourhood].geometry.centroid.coords.xy
                this_bounds = neighbourhood_study_agents[study_neighbourhood].geometry.bounds
                spread_x = int(this_bounds[2] - this_bounds[0])  # Heuristic for agent spread in region
                spread_y = int(this_bounds[3] - this_bounds[1])
                this_x = center_x[0] + self.random.randint(0, spread_x) - spread_x / 2
                this_y = center_y[0] + self.random.randint(0, spread_y) - spread_y / 2
                this_person.activities.append((this_x,this_y))

            # Se le asigna la actividad de trabajo a humanos mayores a 24 años
            elif this_person.age > 24:
                work_neighbourhood = self.random.randint(0, len(neighbourhood_work_agents) - 1)
                center_x, center_y = neighbourhood_work_agents[work_neighbourhood].geometry.centroid.coords.xy
                this_bounds = neighbourhood_work_agents[work_neighbourhood].geometry.bounds
                spread_x = int(this_bounds[2] - this_bounds[0])  # Heuristic for agent spread in region
                spread_y = int(this_bounds[3] - this_bounds[1])
                this_x = center_x[0] + self.random.randint(0, spread_x) - spread_x / 2
                this_y = center_y[0] + self.random.randint(0, spread_y) - spread_y / 2
                this_person.activities.append((this_x,this_y))
            # Se le asigna la actividad de ocio a todos los humanos
            leisure_neighbourhood = self.random.randint(0, len(neighbourhood_leisure_agents) - 1)  # Region where agent starts
            center_x, center_y = neighbourhood_leisure_agents[leisure_neighbourhood].geometry.centroid.coords.xy
            this_bounds = neighbourhood_leisure_agents[leisure_neighbourhood].geometry.bounds
            spread_x = int(this_bounds[2] - this_bounds[0])  # Heuristic for agent spread in region
            spread_y = int(this_bounds[3] - this_bounds[1])
            #this_x = center_x[0] + self.random.randint(0, spread_x) - spread_x / 2
            #this_y = center_y[0] + self.random.randint(0, spread_y) - spread_y / 2
            #this_person.activities.append((this_x,this_y))
            this_person.activities.append((-8408253.832734987, 711215.7371532868))

            # Se cuentan la cantidad de susceptibles e infectados al inicializar
            if this_person.infectionState == "susceptible":
                cont_susc = cont_susc + 1
            elif this_person.infectionState == "infected":
                cont_inf = cont_inf + 1

            # Se agrega el humano creado a el space y al schedule
            self.space.add_agents(this_person)
            self.schedule.add(this_person)

        self.counts["susceptible"] = cont_susc
        self.counts["infected"] = cont_inf

        # Add the neighbourhood agents to schedule AFTER person agents,
        # to allow them to update their color by using BaseScheduler
        for agent in neighbourhood_agents:
            self.schedule.add(agent)

        self.datacollector.collect(self)

    def reset_counts(self):
        self.counts = {
            "susceptible": 0,
            "exposed": 0,
            "infected": 0,
            "recovered": 0,
        }

    def step(self):
        """Run one step of the model."""
        self.steps += 1
        #self.reset_counts()
        self.schedule.step()
        self.space._recreate_rtree()  # Recalculate spatial tree, because agents are moving

        self.datacollector.collect(self) # Calcula la cantidad de susc exp inf y recup

        # Run until simulation time is reached
        if self.steps == self.simulationTime:
            self.running = False

# Functions needed for datacollector
def get_susceptible_count(model):
    return model.counts["susceptible"]


def get_exposed_count(model):
    return model.counts["exposed"]


def get_infected_count(model):
    return model.counts["infected"]


def get_recovered_count(model):
    return model.counts["recovered"]

def generar_booleano(prob):
    booleanos = [True, False]
    probabilidades = [prob, 1-prob]  # Probabilidad de 24% para True y 76% para False
    booleano_aleatorio = random.choices(booleanos, probabilidades)[0]
    return booleano_aleatorio


########################## MY PATCH ##########################
class MyPatch(object):

    # Constructor
    def __init__(self,
                 model,
                 geometry,
                 suceptibleMosquitoes: float,
                 infectedMosquitoes: float,
                 patchType: int,
                 totalMosquitoes0: float,
                 totalMosquitoes1: float):

        # super().__init__(model, geometry)

        # Define constants
        self.naturalEmergenceRate = 0.3
        self.deathRate = 0.071428571428571
        self.mosquitoCarryingCapacity = 1000
        self.mosquitoBiteDemand = 0.5
        self.maxBitesPerHuman = 19
        self.probabilityOfTransmissionHToM = 0.333
        self.probabilityOfTransmissionMToH = 0.333

        # Define constants
        """        
        self.naturalEmergenceRate = 0.3
        self.deathRate = 0.35
        self.mosquitoCarryingCapacity = 317
        self.mosquitoBiteDemand = 0.5
        self.maxBitesPerHuman = 19
        self.probabilityOfTransmissionHToM = 0.87
        self.probabilityOfTransmissionMToH = 0.28
        """
        self.model = model
        self.geometry = geometry
        self.suceptibleMosquitoes = suceptibleMosquitoes
        self.infectedMosquitoes = infectedMosquitoes
        self.patchType = patchType
        # mosquitos totales en los dos tiempos pasados:
        self.totalMosquitoes0 = totalMosquitoes0
        self.totalMosquitoes1 = totalMosquitoes1

    def step(self):
        self.recalculateSEIR()
        susc = self.suceptibleMosquitoes
        inf = self.infectedMosquitoes
        total = susc + inf
        resp = [susc, inf, total]
        return resp

    def recalculateSEIR(self):
        self.solveDiscreteEqns()

    def solveDiscreteEqns(self):
        reproductionRate = 0.12
        tick = int(self.model.steps)
        st = self.suceptibleMosquitoes
        it = self.infectedMosquitoes
        Nv_2 = st + it
        Nv = st + it
        if int(Nv) == 0:
            Nv = 1

        if tick%2 == 0:
            Nv_2 = self.totalMosquitoes1
        else:
            Nv_2 = self.totalMosquitoes0

        Pv = 1 - ((Nv- it)/Nv)**self.maxBitesPerHuman # por notación puede ser Ph
        At = reproductionRate*Nv_2*np.exp((1 - Nv)/self.mosquitoCarryingCapacity)
        s_new = st*(1-self.probabilityOfTransmissionHToM*Pv)*(1-self.deathRate) + At
        i_new = st*self.probabilityOfTransmissionHToM*Pv*(1-self.deathRate) + it*(1-self.deathRate)
        self.suceptibleMosquitoes = int(s_new)
        self.infectedMosquitoes = int(i_new)