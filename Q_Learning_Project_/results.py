import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Plot:

    def __init__(self, carObj, plt_data):
        self.carObj = carObj
        self.plt_data = plt_data

        self.range_step = 1000
        self.collision_bins = np.array(np.zeros([int(self.plt_data.get("steps")/self.range_step)]))
        for collision in self.plt_data.get("collisions"):
            self.collision_bins[int(collision/self.range_step)] += 1


    def plot_(self):

        self.Q_3D_plot()
        self.plot_data_tables_()

        self.plot_collisions_()
        self.plot_space_headway_()
        self.plot_speed_()
        self.plot_relative_speed_()



    def Q_3D_plot(self):
        m = ['+', 'x', '*']
        c = ['b', 'r', 'g']

        plt.ion()
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for d in self.carObj.spaceHeadway_space:
            i_d = self.carObj.spaceHeadway_space.get(d)
            for ds in self.carObj.relativeSpeed_space:
                i_ds = self.carObj.relativeSpeed_space.get(ds)
                for s in self.carObj.speed_space:
                    i_s = self.carObj.speed_space.get(s)
                    a = int(np.argmax(self.carObj.q[i_d, i_ds, i_s]))
                    if not (a == 0 and self.carObj.q[i_d, i_ds, i_s, a] == 0):
                        ax.scatter(d, round(ds * 3.6, 0), round(s * 3.6, 0), marker=m[a], c=c[a])

        ax.set_xlabel('Space headway (m)')
        ax.set_ylabel('Relative Speed (km/h)')
        ax.set_zlabel('Speed (km/h)')

        ax.set_title("3D Q-Table by acceleration")
        ax.legend(handles=[patches.Patch(color='blue', label='acc'),
                           patches.Patch(color='red', label='dec'),
                           patches.Patch(color='green', label='none')],
                  loc=2)

        plt.show()
        plt.draw()
        if (self.carObj.nb_cars>=20):
            plt.savefig("plots/highDensity/3D_Q_Table.jpg")
        else:
            plt.savefig("plots/lowDensity/3D_Q_Table.jpg")
        plt.pause(0.001)

    def plot_collisions_(self):

        plt.ion()
        plt.show()
        plt.style.use('seaborn')
        fig = plt.figure(figsize=(12, 6))

        ax = []

        n_rows = 1
        n_cols = 1

        ax.append(fig.add_subplot(n_rows, n_cols, 1))

        ax[0].bar(np.linspace(0, self.plt_data.get("steps"), len(self.collision_bins)), self.collision_bins, width=1.0,
                  facecolor='r', edgecolor='r')
        ax[0].plot(np.linspace(0, self.plt_data.get("steps"), len(self.collision_bins)), self.collision_bins, 'k--',
                   alpha=0.5)
        ax[0].set_xlabel('Simulation steps')
        ax[0].set_ylabel('Collisions')
        ax[0].set_title("Collisions per 1000 steps", fontsize=10)
        plt.show()
        plt.draw()

        if (self.carObj.nb_cars>=20):
            plt.savefig("plots/highDensity/data_collisions.jpg")
        else:
            plt.savefig("plots/lowDensity/data_collisions.jpg")
        plt.pause(0.001)

    def plot_space_headway_(self):
        plt.ion()
        plt.show()
        plt.style.use('seaborn')
        fig = plt.figure(figsize=(12, 6))
        ax = []
        n_rows = 1
        n_cols = 1

        ax.append(fig.add_subplot(n_rows, n_cols, 1))

        ax[0].plot(np.linspace(0, self.plt_data.get("steps"), len(self.plt_data.get("space_headway"))),
                   self.plt_data.get("space_headway"), 'k', alpha=0.85)
        ax[0].set_xlabel('Simulation steps')
        ax[0].set_ylabel('Space headway (m)')
        ax[0].set_title("Space headway over time", fontsize=10)
        plt.show()
        plt.draw()
        if (self.carObj.nb_cars>=20):
            plt.savefig("plots/highDensity/data_space_headway.jpg")
        else:
            plt.savefig("plots/lowDensity/data_space_headway.jpg")
        plt.pause(0.001)

    def plot_relative_speed_(self):
        plt.ion()
        plt.show()
        plt.style.use('seaborn')
        fig = plt.figure(figsize=(12, 6))
        ax = []
        n_rows = 1
        n_cols = 1

        ax.append(fig.add_subplot(n_rows, n_cols, 1))

        ax[0].plot(np.linspace(0, self.plt_data.get("steps"), len(self.plt_data.get("relative_speed"))),
                   self.plt_data.get("relative_speed"), 'k', alpha=0.85)
        ax[0].set_xlabel('Simulation steps')
        ax[0].set_ylabel('Relative speed (km/h)')
        ax[0].set_title("Relative speed over time", fontsize=10)
        plt.show()
        plt.draw()
        if (self.carObj.nb_cars>=20):
            plt.savefig("plots/highDensity/data_relative_speed.jpg")
        else:
            plt.savefig("plots/lowDensity/data_relative_speed.jpg")
        plt.pause(0.001)

    def plot_speed_(self):
        plt.ion()
        plt.show()
        plt.style.use('seaborn')
        fig = plt.figure(figsize=(12, 6))
        ax = []
        n_rows = 1
        n_cols = 1

        ax.append(fig.add_subplot(n_rows, n_cols, 1))

        ax[0].plot(np.linspace(0, self.plt_data.get("steps"), len(self.plt_data.get("speed"))),
                   self.plt_data.get("speed"), 'k', alpha=0.85)
        ax[0].set_xlabel('Simulation steps')
        ax[0].set_ylabel('Speed (km/h)')
        ax[0].set_title("Speed (km/h) over time", fontsize=10)
        plt.show()
        plt.draw()
        if (self.carObj.nb_cars>=20):
            plt.savefig("plots/highDensity/data_speed.jpg")
        else:
            plt.savefig("plots/lowDensity/data_speed.jpg")

        plt.pause(0.001)

    def plot_data_tables_(self):
        plt.ion()
        plt.show()
        plt.style.use('seaborn')

        fig = plt.figure(figsize=(12, 6))
        fig.suptitle('State variables and number of collisions over time')

        ax = []

        n_rows = 1
        n_cols = 4

        for i in range(4):
            ax.append(fig.add_subplot(n_rows, n_cols, i+1))

        cell_text = []
        total = 0
        for it, b in enumerate(self.collision_bins):
            total += b
            if it < 10 or (it < 50 and (it + 1) % 5 == 0) or (it < 100 and (it + 1) % 10 == 0):
                cell_text.append([(it+1) * self.range_step, int(b), int(total)])

        ax[0].axis('tight')
        ax[0].axis('off')
        ax[0].table(cellText=cell_text, colLabels=("Steps", "Collisions", "Total"), loc='center')
        ax[0].set_title("Collisions", fontsize=10)

        cell_text = []
        space_headway_chunks = [self.plt_data.get("space_headway")[i:i + self.range_step]
                                for i in range(0, len(self.plt_data.get("space_headway")), self.range_step)]

        for it, chunk in enumerate(space_headway_chunks):
            if it < 10 or (it < 50 and (it + 1) % 5 == 0) or (it < 100 and (it + 1) % 10 == 0):
                cell_text.append([(it+1) * self.range_step, round(float(np.mean(chunk)), 2), round(float(np.std(chunk)), 2)])

        ax[1].axis('tight')
        ax[1].axis('off')
        ax[1].table(cellText=cell_text, colLabels=("Steps", "Mean", "Std"), loc='center')
        ax[1].set_title("Space headway (m)", fontsize=10)

        cell_text = []
        relative_speed_chunks = [self.plt_data.get("relative_speed")[i:i + self.range_step]
                                 for i in range(0, len(self.plt_data.get("relative_speed")), self.range_step)]

        for it, chunk in enumerate(relative_speed_chunks):
            if it < 10 or (it < 50 and (it + 1) % 5 == 0) or (it < 100 and (it + 1) % 10 == 0):
                cell_text.append([(it+1) * self.range_step, round(float(np.mean(chunk)), 2), round(float(np.std(chunk)), 2)])

        ax[2].axis('tight')
        ax[2].axis('off')
        ax[2].table(cellText=cell_text, colLabels=("Steps", "Mean", "Std"), loc='center')
        ax[2].set_title("Relative speed (km/h)", fontsize=10)

        cell_text = []
        speed_chunks = [self.plt_data.get("speed")[i:i + self.range_step]
                        for i in range(0, len(self.plt_data.get("speed")), self.range_step)]

        for it, chunk in enumerate(speed_chunks):
            if it < 10 or (it < 50 and (it + 1) % 5 == 0) or (it < 100 and (it + 1) % 10 == 0):
                cell_text.append([(it+1) * self.range_step, round(float(np.mean(chunk)), 2), round(float(np.std(chunk)), 2)])

        ax[3].axis('tight')
        ax[3].axis('off')
        ax[3].table(cellText=cell_text, colLabels=("Steps", "Mean", "Std"), loc='center')
        ax[3].set_title("Speed (km/h)", fontsize=10)

        plt.show()
        plt.draw()
        if (self.carObj.nb_cars>=20):
            plt.savefig("plots/highDensity/data_tables_plot.jpg")
        else:
            plt.savefig("plots/lowDensity/data_tables_plot.jpg")
        plt.pause(0.001)
