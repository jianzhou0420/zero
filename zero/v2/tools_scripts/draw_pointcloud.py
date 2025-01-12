import plotly.graph_objects as go
import numpy as np
import os
import datetime


class PointCloudDrawer:
    def __init__(self):
        pass
    ####################
    ###### Public ######
    ####################

    def draw_onece_blocking(self, points, colors=None, block=True):

        fig = self._draw_onece(points, colors)
        fig.show(block=block)

    def save_onece(self, points, colors=None, save_path=None):
        fig = self._draw_onece(points, colors)

        if save_path is None:
            save_path = os.path.join(os.getcwd(), 'pointclouds')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_path = os.path.join(save_path, f'{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.html')
        fig.write_html(save_path)
        print(f"Pointcloud saved to {save_path}")
        return save_path

    #####################
    ###### private ######
    #####################

    def _draw_onece(self, points, colors):
        x, y, z = np.split(points, 3, axis=1)

        # Generate random RGB colors for each point
        if colors is None:
            colors = np.random.rand(len(x), 3)

        # check if dtype is float

        if issubclass(colors.dtype.type, np.integer):
            colors = colors.astype(float) / 255

        # Create a 3D scatter plot
        fig_data = []
        fig_data.append(go.Scatter3d(x=x.squeeze(), y=y.squeeze(), z=z.squeeze(),
                                     mode='markers',
                                     marker=dict(size=2, color=colors, opacity=0.8)))
        fig = go.Figure(data=fig_data)
        return fig
