import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import norm, chi2, t, weibull_min
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import warnings
from enum import Enum

warnings.filterwarnings('ignore')


class ComponentType(Enum):
    MECHANICAL = "mechanical"
    ELECTRONIC = "electronic"
    POWER = "power"
    HYBRID = "hybrid"


class PerformanceImpact(Enum):
    CRITICAL = "critical"  # Полный отказ системы при неработоспособности
    HIGH = "high"  # Сильное падение производительности
    MEDIUM = "medium"  # Умеренное падение производительности
    LOW = "low"  # Незначительное падение производительности
    REDUNDANT = "redundant"  # Минимальное влияние при наличии резерва


class ServerComponent:
    """Класс для моделирования компонента сервера с учетом влияния на производительность"""

    COMPONENT_DB = {
        'HDD_Enterprise': {
            'mtbf_range': (1000000, 2500000),
            'repair_time': 24,
            'cost': 500,
            'failure_rate_range': (4.0e-7, 1.0e-6),
            'annual_failure_prob': (0.0087, 0.035),
            'component_type': ComponentType.MECHANICAL,
            'weibull_shape': 1.5,
            'common_cause_susceptibility': 0.3,
            'performance_impact': PerformanceImpact.HIGH,
            'performance_weight': 0.15,  # Вклад в общую производительность
            'degradation_profile': 'storage'  # Профиль деградации
        },
        'SSD_Enterprise': {
            'mtbf_range': (2000000, 5000000),
            'repair_time': 12,
            'cost': 1000,
            'failure_rate_range': (2.0e-7, 5.0e-7),
            'annual_failure_prob': (0.0017, 0.0044),
            'component_type': ComponentType.ELECTRONIC,
            'weibull_shape': 1.0,
            'common_cause_susceptibility': 0.1,
            'performance_impact': PerformanceImpact.HIGH,
            'performance_weight': 0.2,
            'degradation_profile': 'storage'
        },
        'Fan_HotSwap': {
            'mtbf_range': (500000, 1000000),
            'repair_time': 4,
            'cost': 100,
            'failure_rate_range': (1.0e-6, 2.0e-6),
            'annual_failure_prob': (0.0088, 0.0175),
            'component_type': ComponentType.MECHANICAL,
            'weibull_shape': 1.8,
            'common_cause_susceptibility': 0.4,
            'performance_impact': PerformanceImpact.MEDIUM,
            'performance_weight': 0.08,
            'degradation_profile': 'cooling'
        },
        'PSU_Redundant': {
            'mtbf_range': (1000000, 1500000),
            'repair_time': 6,
            'cost': 300,
            'failure_rate_range': (6.7e-7, 1.0e-6),
            'annual_failure_prob': (0.0058, 0.0088),
            'component_type': ComponentType.POWER,
            'weibull_shape': 1.2,
            'common_cause_susceptibility': 0.5,
            'performance_impact': PerformanceImpact.CRITICAL,
            'performance_weight': 1.0,  # Критический компонент
            'degradation_profile': 'power'
        },
        'Memory_ECC': {
            'mtbf_range': (2000000, 5000000),
            'repair_time': 8,
            'cost': 400,
            'failure_rate_range': (2.0e-7, 5.0e-7),
            'annual_failure_prob': (0.0017, 0.0044),
            'component_type': ComponentType.ELECTRONIC,
            'weibull_shape': 1.0,
            'common_cause_susceptibility': 0.2,
            'performance_impact': PerformanceImpact.HIGH,
            'performance_weight': 0.25,
            'degradation_profile': 'memory'
        },
        'Motherboard': {
            'mtbf_range': (1500000, 3000000),
            'repair_time': 48,
            'cost': 1500,
            'failure_rate_range': (3.3e-7, 6.7e-7),
            'annual_failure_prob': (0.0029, 0.0058),
            'component_type': ComponentType.HYBRID,
            'weibull_shape': 1.3,
            'common_cause_susceptibility': 0.6,
            'performance_impact': PerformanceImpact.CRITICAL,
            'performance_weight': 1.0,
            'degradation_profile': 'core'
        },
        'CPU': {
            'mtbf_range': (5000000, 10000000),
            'repair_time': 72,
            'cost': 2000,
            'failure_rate_range': (1.0e-7, 2.0e-7),
            'annual_failure_prob': (0.00087, 0.00175),
            'component_type': ComponentType.ELECTRONIC,
            'weibull_shape': 1.0,
            'common_cause_susceptibility': 0.1,
            'performance_impact': PerformanceImpact.CRITICAL,
            'performance_weight': 0.3,
            'degradation_profile': 'compute'
        },
        'Network_Card': {
            'mtbf_range': (1500000, 3000000),
            'repair_time': 12,
            'cost': 200,
            'failure_rate_range': (3.3e-7, 6.7e-7),
            'annual_failure_prob': (0.0029, 0.0058),
            'component_type': ComponentType.ELECTRONIC,
            'weibull_shape': 1.1,
            'common_cause_susceptibility': 0.2,
            'performance_impact': PerformanceImpact.MEDIUM,
            'performance_weight': 0.12,
            'degradation_profile': 'network'
        }
    }

    def __init__(self, name: str, redundancy: int = 1, quality: str = 'typical',
                 environment_factor: float = 1.0):
        if name not in self.COMPONENT_DB:
            raise ValueError(f"Компонент {name} не найден в базе данных")

        self.name = name
        self.db_data = self.COMPONENT_DB[name]
        self.redundancy = redundancy
        self.quality = quality
        self.environment_factor = environment_factor
        self.component_type = self.db_data['component_type']
        self.weibull_shape = self.db_data['weibull_shape']
        self.common_cause_susceptibility = self.db_data['common_cause_susceptibility']
        self.performance_impact = self.db_data['performance_impact']
        self.performance_weight = self.db_data['performance_weight']
        self.degradation_profile = self.db_data['degradation_profile']

        # Выбор конкретных значений в зависимости от качества
        if quality == 'min':
            self.mtbf = self.db_data['mtbf_range'][0]
            self.failure_rate = self.db_data['failure_rate_range'][1]
        elif quality == 'max':
            self.mtbf = self.db_data['mtbf_range'][1]
            self.failure_rate = self.db_data['failure_rate_range'][0]
        else:  # typical
            self.mtbf = np.mean(self.db_data['mtbf_range'])
            self.failure_rate = np.mean(self.db_data['failure_rate_range'])

        # Учет влияния окружающей среды
        self.failure_rate *= environment_factor

        self.repair_time = self.db_data['repair_time']
        self.cost = self.db_data['cost']
        self.annual_failure_prob = np.mean(self.db_data['annual_failure_prob']) * environment_factor

    def calculate_performance_impact(self, failed_instances: int, total_instances: int) -> float:
        """Расчет влияния отказавших экземпляров на производительность"""
        if failed_instances == 0:
            return 1.0  # Полная производительность

        failure_ratio = failed_instances / total_instances

        if self.performance_impact == PerformanceImpact.CRITICAL:
            # Критический компонент - полная потеря производительности при любом отказе
            return 0.0
        elif self.performance_impact == PerformanceImpact.HIGH:
            # Высокое влияние - квадратичная деградация
            return max(0.0, 1.0 - (failure_ratio ** 1.5))
        elif self.performance_impact == PerformanceImpact.MEDIUM:
            # Среднее влияние - линейная деградация
            return 1.0 - failure_ratio
        elif self.performance_impact == PerformanceImpact.LOW:
            # Низкое влияние - логарифмическая деградация
            return 1.0 - (np.log1p(failure_ratio * 10) / np.log1p(10))
        elif self.performance_impact == PerformanceImpact.REDUNDANT:
            # Резервированный - минимальное влияние пока есть рабочие экземпляры
            if failed_instances < total_instances:
                return 1.0 - (failure_ratio * 0.1)  # Только 10% потери за каждый отказавший
            else:
                return 0.0
        else:
            return 1.0 - failure_ratio

    def reliability(self, time: float) -> float:
        """Вероятность безотказной работы за время t с учетом типа компонента"""
        if self.component_type in [ComponentType.MECHANICAL, ComponentType.POWER, ComponentType.HYBRID]:
            scale_param = 1 / self.failure_rate
            return weibull_min.sf(time, self.weibull_shape, scale=scale_param)
        else:
            return np.exp(-self.failure_rate * time)

    def time_to_failure(self) -> float:
        """Генерация времени до отказа с учетом типа распределения"""
        if self.component_type in [ComponentType.MECHANICAL, ComponentType.POWER, ComponentType.HYBRID]:
            scale_param = 1 / self.failure_rate
            return weibull_min.rvs(self.weibull_shape, scale=scale_param)
        else:
            return np.random.exponential(1 / self.failure_rate)

    def availability(self) -> float:
        """Коэффициент готовности компонента"""
        return self.mtbf / (self.mtbf + self.repair_time)

    def redundant_reliability(self, time: float) -> float:
        """Надежность с учетом резервирования и общих причин отказов"""
        if self.redundancy == 1:
            return self.reliability(time)
        else:
            single_reliability = self.reliability(time)
            independent_reliability = 1 - (1 - single_reliability) ** self.redundancy

            common_cause_factor = self.common_cause_susceptibility
            system_reliability = (
                                             1 - common_cause_factor) * independent_reliability + common_cause_factor * single_reliability

            return system_reliability


class ServerArchitecture:
    """Улучшенный класс для моделирования серверной архитектуры с учетом производительности"""

    def __init__(self, name: str, infrastructure_cost: float = 0):
        self.name = name
        self.components = {}
        self.infrastructure_cost = infrastructure_cost
        self.performance_history = []  # История производительности для визуализации

    def add_component(self, component: ServerComponent, quantity: int = 1):
        """Добавление компонента в архитектуру"""
        if component.name in self.components:
            self.components[component.name]['quantity'] += quantity
        else:
            self.components[component.name] = {
                'component': component,
                'quantity': quantity,
                'working_instances': quantity * component.redundancy  # Все экземпляры рабочие изначально
            }

    def calculate_current_performance(self) -> float:
        """Расчет текущей производительности системы на основе состояния компонентов"""
        total_performance = 0.0
        max_possible_performance = 0.0

        for comp_name, comp_data in self.components.items():
            component = comp_data['component']
            total_instances = comp_data['quantity'] * component.redundancy
            working_instances = comp_data['working_instances']
            failed_instances = total_instances - working_instances

            # Расчет влияния на производительность для этого типа компонентов
            performance_factor = component.calculate_performance_impact(failed_instances, total_instances)

            component_performance = component.performance_weight * performance_factor
            total_performance += component_performance
            max_possible_performance += component.performance_weight

        # Нормализация производительности
        if max_possible_performance > 0:
            normalized_performance = total_performance / max_possible_performance
        else:
            normalized_performance = 0.0

        return max(0.0, min(1.0, normalized_performance))

    def simulate_component_failure(self, component_name: str, failed_instances: int = 1):
        """Симуляция отказа компонентов и обновление производительности"""
        if component_name in self.components:
            comp_data = self.components[component_name]
            total_instances = comp_data['quantity'] * comp_data['component'].redundancy
            comp_data['working_instances'] = max(0, comp_data['working_instances'] - failed_instances)

            # Запись в историю производительности
            current_perf = self.calculate_current_performance()
            self.performance_history.append(current_perf)

            return current_perf
        return self.calculate_current_performance()

    def repair_component(self, component_name: str, repaired_instances: int = 1):
        """Ремонт компонентов и восстановление производительности"""
        if component_name in self.components:
            comp_data = self.components[component_name]
            total_instances = comp_data['quantity'] * comp_data['component'].redundancy
            comp_data['working_instances'] = min(total_instances, comp_data['working_instances'] + repaired_instances)

            current_perf = self.calculate_current_performance()
            self.performance_history.append(current_perf)

            return current_perf
        return self.calculate_current_performance()

    def calculate_system_reliability(self, time: float) -> float:
        """Расчет надежности всей системы"""
        system_reliability = 1.0

        for comp_name, comp_data in self.components.items():
            component = comp_data['component']
            quantity = comp_data['quantity']

            if component.redundancy > 1:
                group_reliability = component.redundant_reliability(time)
            else:
                group_reliability = component.reliability(time)

            system_reliability *= group_reliability ** quantity

        return system_reliability

    def calculate_system_availability(self) -> float:
        """Расчет коэффициента готовности системы"""
        system_availability = 1.0

        for comp_name, comp_data in self.components.items():
            component = comp_data['component']
            quantity = comp_data['quantity']

            if component.redundancy > 1:
                comp_unavailability = (1 - component.availability()) ** component.redundancy
                comp_avail = 1 - comp_unavailability
                comp_avail = (1 - component.common_cause_susceptibility) * comp_avail + \
                             component.common_cause_susceptibility * component.availability()
            else:
                comp_avail = component.availability()

            system_availability *= comp_avail ** quantity

        return system_availability

    def calculate_mttf(self) -> float:
        """Расчет среднего времени на отказ системы (MTTF)"""
        system_failure_rate = 0

        for comp_name, comp_data in self.components.items():
            component = comp_data['component']
            quantity = comp_data['quantity']

            if component.redundancy > 1:
                single_failure_rate = component.failure_rate
                mttf_parallel = (1 / single_failure_rate) * sum(1 / i for i in range(1, component.redundancy + 1))
                system_failure_rate += quantity / mttf_parallel
            else:
                system_failure_rate += component.failure_rate * quantity

        return 1 / system_failure_rate if system_failure_rate > 0 else float('inf')

    def annual_failure_probability(self) -> float:
        """Вероятность отказа системы за год"""
        return 1 - self.calculate_system_reliability(8760)

    def monte_carlo_simulation(self, simulation_time: int = 8760, num_simulations: int = 1000) -> Dict:
        """Монте-Карло моделирование с отслеживанием производительности"""
        results = {
            'downtime': [],
            'availability': [],
            'failures': [],
            'operational_time': [],
            'mttf_estimates': [],
            'performance_degradation': [],  # Средняя деградация производительности
            'min_performance': [],  # Минимальная производительность
            'performance_history': []  # Детальная история для одной симуляции
        }

        for sim in range(num_simulations):
            # Сброс состояния системы
            for comp_data in self.components.values():
                comp_data['working_instances'] = comp_data['quantity'] * comp_data['component'].redundancy
            self.performance_history = []

            total_downtime = 0
            total_failures = 0
            failure_times = []
            performance_samples = []
            current_time = 0

            while current_time < simulation_time:
                # Расчет времени до следующего отказа системы
                time_to_failure = self._calculate_system_failure_time()
                failure_time = current_time + time_to_failure

                if failure_time < simulation_time:
                    # Произошел отказ
                    total_failures += 1
                    failure_times.append(time_to_failure)
                    current_time = failure_time

                    # Определяем, какой компонент отказал
                    failed_component = self._select_failed_component()
                    if failed_component:
                        # Симулируем отказ одного экземпляра
                        self.simulate_component_failure(failed_component, 1)

                    # Замеряем производительность после отказа
                    current_perf = self.calculate_current_performance()
                    performance_samples.append(current_perf)

                    # Время восстановления
                    recovery_time = self._calculate_recovery_time()
                    total_downtime += recovery_time
                    current_time += recovery_time

                    # Восстанавливаем все компоненты после ремонта
                    for comp_name in self.components.keys():
                        self.repair_component(comp_name,
                                              self.components[comp_name]['quantity'] *
                                              self.components[comp_name]['component'].redundancy)

                    # Замеряем производительность после восстановления
                    current_perf = self.calculate_current_performance()
                    performance_samples.append(current_perf)
                else:
                    current_time = simulation_time

            # Расчет метрик производительности
            availability = 1 - (total_downtime / simulation_time)
            operational_time = simulation_time - total_downtime

            if failure_times:
                mttf_estimate = np.mean(failure_times)
            else:
                mttf_estimate = simulation_time

            # Статистика по производительности
            if performance_samples:
                avg_performance = np.mean(performance_samples)
                min_performance = np.min(performance_samples) if performance_samples else 1.0
            else:
                avg_performance = 1.0
                min_performance = 1.0

            results['downtime'].append(total_downtime)
            results['availability'].append(availability)
            results['failures'].append(total_failures)
            results['operational_time'].append(operational_time)
            results['mttf_estimates'].append(mttf_estimate)
            results['performance_degradation'].append(1.0 - avg_performance)
            results['min_performance'].append(min_performance)

            # Сохраняем детальную историю только для первой симуляции
            if sim == 0:
                results['performance_history'] = self.performance_history.copy()

        return results

    def _calculate_system_failure_time(self) -> float:
        """Расчет времени до отказа системы"""
        failure_times = []

        for comp_name, comp_data in self.components.items():
            component = comp_data['component']
            quantity = comp_data['quantity']

            for _ in range(quantity):
                if component.redundancy > 1:
                    if np.random.random() < component.common_cause_susceptibility:
                        redundant_failure_time = component.time_to_failure()
                    else:
                        redundant_failure_times = [component.time_to_failure()
                                                   for _ in range(component.redundancy)]
                        redundant_failure_time = np.max(redundant_failure_times)

                    failure_times.append(redundant_failure_time)
                else:
                    failure_time = component.time_to_failure()
                    failure_times.append(failure_time)

        return np.min(failure_times) if failure_times else float('inf')

    def _select_failed_component(self) -> str:
        """Выбор компонента, который отказал (взвешенно по интенсивностям отказов)"""
        components = []
        failure_rates = []

        for comp_name, comp_data in self.components.items():
            component = comp_data['component']
            failure_rates.append(component.failure_rate)
            components.append(comp_name)

        if failure_rates:
            # Вероятность отказа пропорциональна интенсивности отказов
            probabilities = np.array(failure_rates) / np.sum(failure_rates)
            return np.random.choice(components, p=probabilities)
        else:
            return None

    def _calculate_recovery_time(self) -> float:
        """Расчет времени восстановления системы"""
        component_repair_times = []

        for comp_name, comp_data in self.components.items():
            component = comp_data['component']
            quantity = comp_data['quantity']

            for _ in range(quantity):
                repair_time = np.random.lognormal(np.log(component.repair_time), 0.3)
                component_repair_times.append(repair_time)

        return np.median(component_repair_times)

    def sensitivity_analysis(self, parameter_variation: float = 0.1) -> Dict:
        """Анализ чувствительности к изменению параметров компонентов"""
        sensitivity_results = {}
        base_availability = self.calculate_system_availability()
        base_reliability = self.calculate_system_reliability(8760)
        base_performance = self.calculate_current_performance()

        for comp_name, comp_data in self.components.items():
            component = comp_data['component']
            quantity = comp_data['quantity']

            original_failure_rate = component.failure_rate

            # Увеличение интенсивности отказов
            component.failure_rate *= (1 + parameter_variation)
            new_availability_plus = self.calculate_system_availability()
            new_reliability_plus = self.calculate_system_reliability(8760)

            # Уменьшение интенсивности отказов
            component.failure_rate = original_failure_rate * (1 - parameter_variation)
            new_availability_minus = self.calculate_system_availability()
            new_reliability_minus = self.calculate_system_reliability(8760)

            # Восстановление исходного значения
            component.failure_rate = original_failure_rate

            availability_sensitivity = (new_availability_plus - new_availability_minus) / (
                        2 * parameter_variation * base_availability)
            reliability_sensitivity = (new_reliability_plus - new_reliability_minus) / (
                        2 * parameter_variation * base_reliability)

            sensitivity_results[comp_name] = {
                'availability_sensitivity': availability_sensitivity,
                'reliability_sensitivity': reliability_sensitivity,
                'impact_level': 'high' if abs(availability_sensitivity) > 0.1 else
                'medium' if abs(availability_sensitivity) > 0.05 else 'low'
            }

        return sensitivity_results


class ResilienceAnalyzer:
    """Анализатор устойчивости серверной архитектуры с анализом производительности"""

    def __init__(self):
        self.architectures = {}

    def add_architecture(self, architecture: ServerArchitecture):
        """Добавление архитектуры для анализа"""
        self.architectures[architecture.name] = architecture

    def compare_architectures(self, time_horizon: int = 8760) -> pd.DataFrame:
        """Сравнительный анализ архитектур с метриками производительности"""
        comparison_data = []

        for arch_name, architecture in self.architectures.items():
            # Базовые метрики
            reliability = architecture.calculate_system_reliability(time_horizon)
            availability = architecture.calculate_system_availability()
            annual_failure_prob = architecture.annual_failure_probability()
            mttf = architecture.calculate_mttf()

            # Монте-Карло симуляция
            mc_results = architecture.monte_carlo_simulation(time_horizon, 500)
            avg_availability = np.mean(mc_results['availability'])
            avg_downtime = np.mean(mc_results['downtime'])
            avg_failures = np.mean(mc_results['failures'])
            avg_mttf_sim = np.mean(mc_results['mttf_estimates'])
            avg_performance_degradation = np.mean(mc_results['performance_degradation'])
            min_performance = np.mean(mc_results['min_performance'])

            # Анализ чувствительности
            sensitivity = architecture.sensitivity_analysis()
            high_sensitivity_components = sum(1 for s in sensitivity.values()
                                              if s['impact_level'] == 'high')

            # Стоимость системы
            total_cost = self._calculate_total_cost(architecture)

            comparison_data.append({
                'Архитектура': arch_name,
                'Надежность': reliability,
                'Готовность_Теор': availability,
                'Готовность_Сим': avg_availability,
                'MTTF_Теор_ч': mttf,
                'MTTF_Сим_ч': avg_mttf_sim,
                'Вероятность_Отказа_Год': annual_failure_prob,
                'Простой_Часов_Год': avg_downtime,
                'Отказов_Год': avg_failures,
                'Деградация_Производительности': avg_performance_degradation,
                'Мин_Производительность': min_performance,
                'Критичные_Компоненты': high_sensitivity_components,
                'Стоимость': total_cost,
                'Эффективность': availability / total_cost * 1e6 if total_cost > 0 else 0
            })

        return pd.DataFrame(comparison_data)

    def _calculate_total_cost(self, architecture: ServerArchitecture) -> float:
        """Расчет общей стоимости архитектуры с учетом инфраструктуры"""
        total_cost = architecture.infrastructure_cost

        for comp_name, comp_data in architecture.components.items():
            component = comp_data['component']
            quantity = comp_data['quantity']

            component_cost = component.cost * quantity * component.redundancy
            maintenance_cost = component_cost * 0.2
            total_cost += component_cost + maintenance_cost

        return total_cost

    def plot_performance_degradation(self, architecture_name: str, simulation_time: int = 8760):
        """Визуализация деградации производительности во времени"""
        if architecture_name not in self.architectures:
            print(f"Архитектура {architecture_name} не найдена")
            return

        architecture = self.architectures[architecture_name]

        # Запускаем детальную симуляцию
        mc_results = architecture.monte_carlo_simulation(simulation_time, 1)
        performance_history = mc_results['performance_history']

        if not performance_history:
            print("Нет данных о производительности")
            return

        plt.figure(figsize=(12, 6))

        # Создаем временную шкалу
        time_points = np.linspace(0, simulation_time, len(performance_history))

        plt.plot(time_points, performance_history, 'b-', alpha=0.7, linewidth=1)
        plt.fill_between(time_points, performance_history, alpha=0.3, color='blue')

        plt.xlabel('Время (часы)')
        plt.ylabel('Производительность системы')
        plt.title(f'Деградация производительности во времени\n{architecture_name}')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)

        # Добавляем критические уровни производительности
        plt.axhline(y=0.8, color='orange', linestyle='--', alpha=0.7, label='Высокая нагрузка (80%)')
        plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Критическая нагрузка (50%)')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_component_performance_impact(self, architecture_name: str):
        """Анализ влияния компонентов на производительность"""
        if architecture_name not in self.architectures:
            return

        architecture = self.architectures[architecture_name]

        components = []
        performance_weights = []
        performance_impacts = []
        redundancy_levels = []

        for comp_name, comp_data in architecture.components.items():
            component = comp_data['component']
            quantity = comp_data['quantity']

            components.append(f"{comp_name}\n(x{quantity})")
            performance_weights.append(component.performance_weight * 100)  # в процентах
            performance_impacts.append(component.performance_impact.value)
            redundancy_levels.append(component.redundancy)

        # Создаем цветовую карту для типов влияния
        impact_colors = {
            'critical': 'red',
            'high': 'orange',
            'medium': 'yellow',
            'low': 'green',
            'redundant': 'blue'
        }
        colors = [impact_colors[impact] for impact in performance_impacts]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # График весов производительности
        bars1 = ax1.bar(components, performance_weights, color=colors, alpha=0.7)
        ax1.set_title('Вклад компонентов в производительность системы')
        ax1.set_ylabel('Вес производительности (%)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)

        # Добавляем аннотации с уровнем резервирования
        for i, (bar, redundancy) in enumerate(zip(bars1, redundancy_levels)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 1,
                     f'R:{redundancy}', ha='center', va='bottom', fontsize=8)

        # График уровней влияния
        impact_counts = {}
        for impact in performance_impacts:
            impact_counts[impact] = impact_counts.get(impact, 0) + 1

        ax2.pie(impact_counts.values(), labels=impact_counts.keys(),
                colors=[impact_colors[impact] for impact in impact_counts.keys()],
                autopct='%1.1f%%', startangle=90)
        ax2.set_title('Распределение компонентов по уровню влияния')

        plt.tight_layout()
        plt.show()

    def plot_reliability_comparison(self, max_time: int = 20000):
        """Построение графиков сравнения надежности"""
        plt.figure(figsize=(14, 8))

        time_points = np.linspace(1, max_time, 100)

        for arch_name, architecture in self.architectures.items():
            reliability_curve = [architecture.calculate_system_reliability(t) for t in time_points]
            plt.plot(time_points / 8760, reliability_curve, label=arch_name, linewidth=2)

        plt.xlabel('Время (годы)')
        plt.ylabel('Вероятность безотказной работы')
        plt.title('Сравнение надежности серверных архитектур\n(с учетом влияния на производительность)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        plt.xlim(0, max_time / 8760)
        plt.show()


def create_realistic_architectures() -> ResilienceAnalyzer:
    """Создание реалистичных архитектур с учетом производительности"""

    analyzer = ResilienceAnalyzer()

    # 1. Базовая архитектура (минимальная надежность)
    basic = ServerArchitecture("Базовая (мин.надежность)", infrastructure_cost=5000)
    basic.add_component(ServerComponent("HDD_Enterprise", redundancy=1, quality='min'), quantity=4)
    basic.add_component(ServerComponent("PSU_Redundant", redundancy=1, quality='min'), quantity=2)
    basic.add_component(ServerComponent("Fan_HotSwap", redundancy=1, quality='min'), quantity=6)
    basic.add_component(ServerComponent("Memory_ECC", redundancy=1, quality='min'), quantity=8)
    basic.add_component(ServerComponent("Motherboard", redundancy=1, quality='min'), quantity=1)
    basic.add_component(ServerComponent("CPU", redundancy=1, quality='min'), quantity=2)
    basic.add_component(ServerComponent("Network_Card", redundancy=1, quality='min'), quantity=1)

    # 2. Сбалансированная архитектура
    balanced = ServerArchitecture("Сбалансированная", infrastructure_cost=8000)
    balanced.add_component(ServerComponent("SSD_Enterprise", redundancy=1, quality='typical'), quantity=4)
    balanced.add_component(ServerComponent("PSU_Redundant", redundancy=2, quality='typical'), quantity=2)
    balanced.add_component(ServerComponent("Fan_HotSwap", redundancy=2, quality='typical'), quantity=8)
    balanced.add_component(ServerComponent("Memory_ECC", redundancy=1, quality='typical'), quantity=16)
    balanced.add_component(ServerComponent("Motherboard", redundancy=1, quality='typical'), quantity=1)
    balanced.add_component(ServerComponent("CPU", redundancy=1, quality='typical'), quantity=2)
    balanced.add_component(ServerComponent("Network_Card", redundancy=2, quality='typical'), quantity=1)

    # 3. Высоконадежная архитектура
    high_avail = ServerArchitecture("Высоконадежная", infrastructure_cost=15000)
    high_avail.add_component(ServerComponent("SSD_Enterprise", redundancy=2, quality='max'), quantity=8)
    high_avail.add_component(ServerComponent("PSU_Redundant", redundancy=3, quality='max'), quantity=2)
    high_avail.add_component(ServerComponent("Fan_HotSwap", redundancy=3, quality='max'), quantity=12)
    high_avail.add_component(ServerComponent("Memory_ECC", redundancy=2, quality='max'), quantity=16)
    high_avail.add_component(ServerComponent("Motherboard", redundancy=2, quality='max'), quantity=2)
    high_avail.add_component(ServerComponent("CPU", redundancy=1, quality='max'), quantity=4)
    high_avail.add_component(ServerComponent("Network_Card", redundancy=2, quality='max'), quantity=2)

    # 4. Экономичная архитектура
    eco = ServerArchitecture("Экономичная", infrastructure_cost=3000)
    eco.add_component(ServerComponent("HDD_Enterprise", redundancy=1, quality='min'), quantity=2)
    eco.add_component(ServerComponent("PSU_Redundant", redundancy=1, quality='min'), quantity=1)
    eco.add_component(ServerComponent("Fan_HotSwap", redundancy=1, quality='min'), quantity=4)
    eco.add_component(ServerComponent("Memory_ECC", redundancy=1, quality='min'), quantity=4)
    eco.add_component(ServerComponent("Motherboard", redundancy=1, quality='min'), quantity=1)
    eco.add_component(ServerComponent("CPU", redundancy=1, quality='min'), quantity=1)
    eco.add_component(ServerComponent("Network_Card", redundancy=1, quality='min'), quantity=1)

    analyzer.add_architecture(basic)
    analyzer.add_architecture(balanced)
    analyzer.add_architecture(high_avail)
    analyzer.add_architecture(eco)

    return analyzer


def demonstrate_performance_degradation():
    """Демонстрация падения производительности при отказах"""
    print("=== ДЕМОНСТРАЦИЯ ВЛИЯНИЯ ОТКАЗОВ НА ПРОИЗВОДИТЕЛЬНОСТЬ ===\n")

    # Создаем тестовую архитектуру
    test_arch = ServerArchitecture("Тестовая система")
    test_arch.add_component(ServerComponent("CPU", redundancy=2, quality='typical'), quantity=2)
    test_arch.add_component(ServerComponent("Memory_ECC", redundancy=1, quality='typical'), quantity=8)
    test_arch.add_component(ServerComponent("SSD_Enterprise", redundancy=2, quality='typical'), quantity=4)
    test_arch.add_component(ServerComponent("PSU_Redundant", redundancy=2, quality='typical'), quantity=2)

    print("Начальное состояние системы:")
    initial_perf = test_arch.calculate_current_performance()
    print(f"Производительность: {initial_perf:.1%}")

    # Симулируем серию отказов
    failures = [
        ("CPU", 1, "Отказ одного процессора"),
        ("Memory_ECC", 2, "Отказ двух модулей памяти"),
        ("SSD_Enterprise", 1, "Отказ одного SSD"),
        ("PSU_Redundant", 1, "Отказ одного блока питания"),
        ("CPU", 1, "Отказ второго процессора (критический)"),
    ]

    print("\nСимуляция отказов:")
    print("-" * 60)

    for component, count, description in failures:
        before_perf = test_arch.calculate_current_performance()
        after_perf = test_arch.simulate_component_failure(component, count)
        degradation = before_perf - after_perf

        print(f"{description}:")
        print(f"  Производительность: {before_perf:.1%} → {after_perf:.1%}")
        print(f"  Падение: {degradation:.1%}")

        if after_perf == 0:
            print("  🚨 СИСТЕМА ПОЛНОСТЬЮ НЕРАБОТОСПОСОБНА")
            break

    print("\nВосстановление системы...")
    # Восстанавливаем все компоненты
    for comp_name in test_arch.components.keys():
        test_arch.repair_component(comp_name,
                                   test_arch.components[comp_name]['quantity'] *
                                   test_arch.components[comp_name]['component'].redundancy)

    final_perf = test_arch.calculate_current_performance()
    print(f"Производительность после восстановления: {final_perf:.1%}")


def main():
    """Основная функция демонстрации системы с учетом производительности"""
    print("=== СИСТЕМА МОДЕЛИРОВАНИЯ УСТОЙЧИВОСТИ С УЧЕТОМ ПРОИЗВОДИТЕЛЬНОСТИ ===\n")

    # Демонстрация влияния отказов на производительность
    demonstrate_performance_degradation()

    print("\n" + "=" * 100)
    print("ПОЛНЫЙ АНАЛИЗ АРХИТЕКТУР")
    print("=" * 100)

    # Создание и анализ архитектур
    analyzer = create_realistic_architectures()

    # 1. Сравнительный анализ
    print("\n1. СРАВНИТЕЛЬНЫЙ АНАЛИЗ АРХИТЕКТУР:")
    comparison_df = analyzer.compare_architectures()
    pd.set_option('display.float_format', '{:.6f}'.format)
    pd.set_option('display.max_columns', None)
    print(comparison_df.to_string(index=False))

    # 2. Графики производительности
    print("\n2. АНАЛИЗ ПРОИЗВОДИТЕЛЬНОСТИ:")

    # Графики деградации производительности для каждой архитектуры
    for arch_name in analyzer.architectures.keys():
        print(f"\nАнализ производительности: {arch_name}")
        analyzer.plot_performance_degradation(arch_name, 8760)  # 1 год
        analyzer.plot_component_performance_impact(arch_name)

    # 3. Сравнение надежности
    print("\n3. СРАВНЕНИЕ НАДЕЖНОСТИ:")
    analyzer.plot_reliability_comparison()

    # 4. Детальный анализ
    print("\n4. ДЕТАЛЬНЫЙ АНАЛИЗ РЕЗУЛЬТАТОВ:")

    best_reliability = comparison_df.loc[comparison_df['Надежность'].idxmax()]
    best_availability = comparison_df.loc[comparison_df['Готовность_Сим'].idxmax()]
    best_performance = comparison_df.loc[comparison_df['Мин_Производительность'].idxmax()]
    best_value = comparison_df.loc[comparison_df['Эффективность'].idxmax()]

    print(f"Самая надежная архитектура: {best_reliability['Архитектура']}")
    print(f"  - Надежность: {best_reliability['Надежность']:.4f}")
    print(f"  - Мин. производительность: {best_reliability['Мин_Производительность']:.1%}")

    print(f"\nАрхитектура с лучшей готовностью: {best_availability['Архитектура']}")
    print(f"  - Готовность: {best_availability['Готовность_Сим']:.4f}")
    print(f"  - Простой: {best_availability['Простой_Часов_Год']:.1f} часов/год")

    print(f"\nАрхитектура с лучшей производительностью: {best_performance['Архитектура']}")
    print(f"  - Мин. производительность: {best_performance['Мин_Производительность']:.1%}")
    print(f"  - Средняя деградация: {best_performance['Деградация_Производительности']:.1%}")

    print(f"\nНаиболее эффективная по стоимости: {best_value['Архитектура']}")
    print(f"  - Стоимость: ${best_value['Стоимость']:,.0f}")
    print(f"  - Эффективность: {best_value['Эффективность']:.2f}")

    # 5. Рекомендации по оптимизации
    print("\n5. РЕКОМЕНДАЦИИ ПО ОПТИМИЗАЦИИ ПРОИЗВОДИТЕЛЬНОСТИ:")
    print("-" * 60)

    for arch_name, architecture in analyzer.architectures.items():
        print(f"\n{arch_name}:")

        # Анализ критических компонентов
        critical_components = []
        for comp_name, comp_data in architecture.components.items():
            component = comp_data['component']
            if component.performance_impact in [PerformanceImpact.CRITICAL, PerformanceImpact.HIGH]:
                critical_components.append((comp_name, component.performance_impact.value))

        if critical_components:
            print("  Критические компоненты для производительности:")
            for comp_name, impact in critical_components:
                print(f"    - {comp_name} ({impact})")
        else:
            print("  Нет критических компонентов")

        # Рекомендации по резервированию
        low_redundancy = []
        for comp_name, comp_data in architecture.components.items():
            component = comp_data['component']
            if (component.performance_impact in [PerformanceImpact.CRITICAL, PerformanceImpact.HIGH]
                    and component.redundancy == 1):
                low_redundancy.append(comp_name)

        if low_redundancy:
            print("  Рекомендуется увеличить резервирование:")
            for comp_name in low_redundancy:
                print(f"    - {comp_name}")


if __name__ == "__main__":
    main()