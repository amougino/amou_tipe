import thren_v_alpha as thren


tau = (0, 500000)
prec = 10000

settings = thren.get_single_settings()
a = thren.calculate(settings, tau)

thren.plot_traj(a, settings, tau, prec)
