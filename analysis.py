import math


def lumped_temperature(t, mass, cp, area, h, ambient, initial, power):
    capacity = mass * cp
    conductance = max(0.0, h * area)
    if conductance < 1e-15:
        return initial + power * t / capacity
    e = math.exp(-conductance * t / capacity)
    return ambient + power / conductance + (initial - ambient - power / conductance) * e


def fit_h(request):
    points = sorted(request.measurements, key=lambda x: x.time_s)
    low, high = sorted(request.h_bounds)
    for _ in range(90):
        mid = (low + high) / 2
        # A one-parameter least-squares search is robust for noisy temperature data.
        delta = max(mid * 1e-4, 1e-6)
        left = sum((lumped_temperature(p.time_s, request.mass_kg, request.cp_J_kgK, request.area_m2, mid - delta, request.ambient_C, request.initial_C, request.power_W) - p.temperature_C) ** 2 for p in points)
        right = sum((lumped_temperature(p.time_s, request.mass_kg, request.cp_J_kgK, request.area_m2, mid + delta, request.ambient_C, request.initial_C, request.power_W) - p.temperature_C) ** 2 for p in points)
        if left < right:
            high = mid
        else:
            low = mid
    h = (low + high) / 2
    residual = math.sqrt(sum((lumped_temperature(p.time_s, request.mass_kg, request.cp_J_kgK, request.area_m2, h, request.ambient_C, request.initial_C, request.power_W) - p.temperature_C) ** 2 for p in points) / len(points))
    return dict(mode='fit_h', h_W_m2K=h, rmse_C=residual, predictions=[dict(time_s=p.time_s, measured_C=p.temperature_C, predicted_C=lumped_temperature(p.time_s, request.mass_kg, request.cp_J_kgK, request.area_m2, h, request.ambient_C, request.initial_C, request.power_W)) for p in points])


def power_limit(request):
    h = request.h_W_m2K
    low, high = 0.0, max(request.power_W, 1.0)
    while lumped_temperature(request.duration_s, request.mass_kg, request.cp_J_kgK, request.area_m2, h, request.ambient_C, request.initial_C, high) < request.max_temperature_C and high < 1e12:
        high *= 2
    for _ in range(90):
        mid = (low + high) / 2
        if lumped_temperature(request.duration_s, request.mass_kg, request.cp_J_kgK, request.area_m2, h, request.ambient_C, request.initial_C, mid) <= request.max_temperature_C:
            low = mid
        else:
            high = mid
    power = low
    return dict(mode='power_limit', h_W_m2K=h, maximum_power_W=power, predicted_temperature_C=lumped_temperature(request.duration_s, request.mass_kg, request.cp_J_kgK, request.area_m2, h, request.ambient_C, request.initial_C, power), duration_s=request.duration_s)


def run_calibration(request):
    return fit_h(request) if request.mode == 'fit_h' else power_limit(request)


def run_cycle(request):
    te = request.evaporating_C + 273.15
    tc = request.condensing_C + 273.15
    carnot = te / (tc - te)
    cop = carnot * request.compressor_efficiency
    compressor_power = request.cooling_capacity_W / cop
    total_power = compressor_power + request.fan_pump_power_W
    return dict(refrigerant=request.refrigerant, evaporating_C=request.evaporating_C, condensing_C=request.condensing_C, suction_C=request.evaporating_C + request.superheat_C, liquid_C=request.condensing_C - request.subcool_C, cooling_capacity_W=request.cooling_capacity_W, compressor_power_W=compressor_power, auxiliary_power_W=request.fan_pump_power_W, total_power_W=total_power, heating_rejection_W=request.cooling_capacity_W + compressor_power, cop=cop, eer_Btu_Wh=cop * 3.412142, model='理想循环温差估算；未调用工质物性库')
