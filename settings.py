import tkinter as tk
from tkinter import ttk


class SettingsWindow:
    def __init__(self, initial_speed, initial_dpi, initial_cursor_size, callback):
        """
        Inicializa a janela de configurações do mouse.
        :param initial_speed: Velocidade inicial do mouse.
        :param initial_dpi: DPI inicial do mouse.
        :param initial_cursor_size: Tamanho inicial do cursor.
        :param callback: Função a ser chamada ao aplicar configurações.
        """
        self.initial_speed = initial_speed
        self.initial_dpi = initial_dpi
        self.initial_cursor_size = initial_cursor_size
        self.callback = callback

        self.window = tk.Tk()
        self.window.title("Configurações do Mouse")

        # Velocidade do Mouse
        ttk.Label(self.window, text="Velocidade do Mouse:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.speed_var = tk.StringVar(value=str(self.initial_speed))
        ttk.Entry(self.window, textvariable=self.speed_var).grid(row=0, column=1, padx=10, pady=5)

        # DPI do Mouse
        ttk.Label(self.window, text="DPI do Mouse:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.dpi_var = tk.StringVar(value=str(self.initial_dpi))
        ttk.Entry(self.window, textvariable=self.dpi_var).grid(row=1, column=1, padx=10, pady=5)

        # Tamanho do Cursor
        ttk.Label(self.window, text="Tamanho do Cursor:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.cursor_size_var = tk.StringVar(value=str(self.initial_cursor_size))
        ttk.Entry(self.window, textvariable=self.cursor_size_var).grid(row=2, column=1, padx=10, pady=5)

        # Botão de Aplicar
        ttk.Button(self.window, text="Aplicar", command=self.apply_settings).grid(row=3, column=0, columnspan=2, pady=10)

    def apply_settings(self):
        """
        Aplica as configurações e chama o callback.
        """
        new_speed = float(self.speed_var.get())
        new_dpi = float(self.dpi_var.get())
        new_cursor_size = int(self.cursor_size_var.get())

        # Chama o callback com os novos valores
        self.callback(new_speed, new_dpi, new_cursor_size)

        # Fecha a janela
        self.window.destroy()

    def show(self):
        """
        Mostra a janela do Tkinter.
        """
        self.window.mainloop()
