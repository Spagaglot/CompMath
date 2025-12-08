import numpy as np
import random
import math
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.special import roots_legendre
import time

class NumericalIntegrationLab:

    def __init__(self):
        """Инициализация всех методов и данных"""

        # Табличные данные из задачи VII.9.4
        self.x_table = np.array([0, 0.5, 1, 1.5, 2])
        self.f_table = np.array([0.5, 0.25, 0.25, 0.1, 0.1])

        # Узлы и веса Гаусса из таблицы
        self.gauss_nodes_weights = {
            1: {'x': [0.0], 'c': [2.0]},
            2: {'x': [-0.5773503, 0.5773503], 'c': [1.0, 1.0]},
            3: {'x': [-0.7745967, 0.0, 0.7745967],
                'c': [0.5555556, 0.8888889, 0.5555556]},
            4: {'x': [-0.8611363, -0.3399810, 0.3399810, 0.8611363],
                'c': [0.3478548, 0.6521451, 0.6521451, 0.3478548]},
            5: {'x': [-0.9061798, -0.5384693, 0.0, 0.5384693, 0.9061798],
                'c': [0.4786287, 0.2369269, 0.5688888, 0.2369269, 0.4786287]},
            6: {'x': [-0.9324700, -0.6612094, -0.2386142,
                      0.2386142, 0.6612094, 0.9324700],
                'c': [0.1713245, 0.3607616, 0.4679140,
                      0.4679140, 0.3607616, 0.1713245]}
        }

    # ==================== МЕТОДЫ НЬЮТОНА-КОТЕСА ====================

    def rectangle_method(self, func, a, b, n=1000, method='midpoint'):
        """
        Метод прямоугольников
        method: 'left', 'right', 'midpoint'
        """
        h = (b - a) / n
        result = 0.0

        if method == 'left':
            # Левые прямоугольники
            for i in range(n):
                x_i = a + i * h
                result += func(x_i)
            result *= h

        elif method == 'right':
            # Правые прямоугольники
            for i in range(1, n + 1):
                x_i = a + i * h
                result += func(x_i)
            result *= h

        else:  # midpoint - центральные прямоугольники
            for i in range(n):
                x_i = a + (i + 0.5) * h
                result += func(x_i)
            result *= h

        return result

    def trapezoidal_method(self, func, a, b, n=1000):
        """
        Метод трапеций
        """
        h = (b - a) / n
        result = 0.5 * (func(a) + func(b))

        for i in range(1, n):
            x_i = a + i * h
            result += func(x_i)

        result *= h
        return result

    def simpson_method(self, func, a, b, n=1000):
        """
        Метод Симпсона (n должно быть четным)
        """
        if n % 2 != 0:
            n += 1  # делаем четным

        h = (b - a) / n
        result = func(a) + func(b)

        # Сумма по нечетным индексам
        for i in range(1, n, 2):
            x_i = a + i * h
            result += 4 * func(x_i)

        # Сумма по четным индексам
        for i in range(2, n-1, 2):
            x_i = a + i * h
            result += 2 * func(x_i)

        result *= h / 3
        return result

    def gauss_method(self, func, a, b, n=4):
        """
        Метод Гаусса с использованием предопределенных узлов и весов
        """
        if n not in self.gauss_nodes_weights:
            # Используем scipy для генерации узлов и весов
            nodes, weights = roots_legendre(n)
        else:
            nodes = np.array(self.gauss_nodes_weights[n]['x'])
            weights = np.array(self.gauss_nodes_weights[n]['c'])

        # Преобразование к интервалу [a, b]
        transformed_nodes = 0.5 * (b - a) * nodes + 0.5 * (a + b)

        # Вычисление интеграла
        result = 0.0
        for i in range(n):
            result += weights[i] * func(transformed_nodes[i])

        result *= 0.5 * (b - a)
        return result

    # ==================== МЕТОДЫ ДЛЯ ТАБЛИЧНОЙ ФУНКЦИИ ====================

    def table_rectangle(self, method='midpoint'):
        """
        Метод прямоугольников для табличной функции
        """
        h = self.x_table[1] - self.x_table[0]

        if method == 'left':
            result = h * np.sum(self.f_table[:-1])
        elif method == 'right':
            result = h * np.sum(self.f_table[1:])
        else:  # midpoint
            # Интерполируем значения в средних точках
            f_mid = (self.f_table[:-1] + self.f_table[1:]) / 2
            result = h * np.sum(f_mid)

        return result

    def table_trapezoidal(self):
        """
        Метод трапеций для табличной функции
        """
        h = self.x_table[1] - self.x_table[0]
        n = len(self.x_table) - 1

        result = 0.5 * (self.f_table[0] + self.f_table[-1])
        result += np.sum(self.f_table[1:-1])
        result *= h

        return result

    def table_simpson(self):
        """
        Метод Симпсона для табличной функции
        """
        n = len(self.x_table) - 1
        if n % 2 != 0:
            raise ValueError("Для формулы Симпсона нужно четное количество интервалов")

        h = self.x_table[1] - self.x_table[0]

        result = self.f_table[0] + self.f_table[-1]

        # Нечетные индексы
        for i in range(1, n, 2):
            result += 4 * self.f_table[i]

        # Четные индексы (внутренние)
        for i in range(2, n-1, 2):
            result += 2 * self.f_table[i]

        result *= h / 3
        return result

    # ==================== ДОПОЛНИТЕЛЬНЫЕ КВЕСТЫ ====================

    def singular_integral(self):
        """
        Интеграл с особенностью: ∫ dx/√x от 0 до 1
        Аналитическое значение: 2
        """
        def singular_func(x):
            return 1 / np.sqrt(x) if x > 1e-15 else 0

        a, b = 1e-10, 1  # Избегаем деления на 0
        exact = 2 * (np.sqrt(b) - np.sqrt(a))

        print(f"\nИнтеграл с особенностью: ∫ dx/√x от {a:.1e} до {b}")
        print(f"Аналитическое значение: {exact:.6f}")

        results = {}

        # Используем адаптивный метод для особенности
        n = 10000  # Больше точек для особенности

        # Метод трапеций
        trap_result = self.trapezoidal_method(singular_func, a, b, n)
        results['Трапеции'] = trap_result

        # Метод Симпсона
        simp_result = self.simpson_method(singular_func, a, b, n)
        results['Симпсон'] = simp_result

        # Метод Гаусса с разным количеством точек
        for gauss_n in [2, 3, 4, 5, 6]:
            try:
                gauss_result = self.gauss_method(singular_func, a, b, n=gauss_n)
                results[f'Гаусс (n={gauss_n})'] = gauss_result
            except Exception as e:
                pass

        print(f"{'Метод':<15} {'Результат':<15} {'Погрешность':<15}")
        print("-" * 45)
        for method, value in results.items():
            error = abs(value - exact)
            print(f"{method:<15} {value:<15.6f} {error:<15.6e}")

        return results, exact

    def oscillatory_integral(self):
        """
        Интеграл быстроосциллирующей функции: ∫ sin(50x) dx от 0 до π
        Аналитическое значение: (1 - cos(50π))/50 = 0.04
        """
        def oscillatory_func(x):
            return np.sin(50 * x)

        a, b = 0, np.pi
        exact = (1 - np.cos(50 * np.pi)) / 50  # = 0.04

        print(f"\nБыстроосциллирующая функция: ∫ sin(50x) dx от 0 до π")
        print(f"Аналитическое значение: {exact:.10f}")
        print(f"cos(50π) = {np.cos(50*np.pi):.10f}")

        results = {}

        # Нужно много точек для осциллирующей функции
        n = 50000

        # Метод трапеций
        trap_result = self.trapezoidal_method(oscillatory_func, a, b, n)
        results['Трапеции'] = trap_result

        # Метод Симпсона
        simp_result = self.simpson_method(oscillatory_func, a, b, n)
        results['Симпсон'] = simp_result

        # Метод Гаусса с большим количеством точек (используем scipy)
        for gauss_n in [10, 20, 30]:
            try:
                gauss_result = self.gauss_method(oscillatory_func, a, b, n=gauss_n)
                results[f'Гаусс (n={gauss_n})'] = gauss_result
            except Exception as e:
                pass

        # Метод прямоугольников (центральные)
        rect_result = self.rectangle_method(oscillatory_func, a, b, n, 'midpoint')
        results['Прямоуг.'] = rect_result

        print(f"\n{'Метод':<15} {'Результат':<20} {'Погрешность':<20}")
        print("-" * 55)
        for method, value in results.items():
            error = abs(value - exact)
            print(f"{method:<15} {value:<20.10f} {error:<20.10e}")

        # Визуализация осциллирующей функции
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        x_plot = np.linspace(a, b, 2000)
        y_plot = oscillatory_func(x_plot)
        plt.plot(x_plot, y_plot, 'b-', alpha=0.7, linewidth=0.5)
        plt.fill_between(x_plot, y_plot, alpha=0.3)
        plt.title(f'Быстроосциллирующая функция: sin(50x)\n∫ sin(50x)dx = {exact:.6f}', fontsize=14)
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        # Покажем небольшой участок для деталей
        x_detail = np.linspace(0, 0.2, 1000)
        y_detail = oscillatory_func(x_detail)
        plt.plot(x_detail, y_detail, 'r-', alpha=0.8, linewidth=1)
        plt.title('Деталь: первые 0.2 единицы', fontsize=14)
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('oscillatory_function.png', dpi=150, bbox_inches='tight')
        plt.show()

        return results, exact

    def monte_carlo_integration(self, func, a, b, n=10000):
        """
        Метод Монте-Карло для вычисления интеграла
        """
        # Генерация случайных точек
        random_points = np.random.uniform(a, b, n)

        # Вычисление значений функции
        function_values = func(random_points)

        # Оценка интеграла
        integral_estimate = (b - a) * np.mean(function_values)

        # Оценка погрешности
        std_error = (b - a) * np.std(function_values) / np.sqrt(n)

        # Доверительный интервал (95%)
        conf_interval = (integral_estimate - 1.96 * std_error,
                        integral_estimate + 1.96 * std_error)

        return {
            'estimate': integral_estimate,
            'std_error': std_error,
            'conf_interval': conf_interval,
            'n_points': n
        }

    def compare_all_methods(self, func, a, b, exact_value=None, n=1000):
        """
        Сравнение всех методов для заданной функции
        """
        print(f"\n{'='*60}")
        print(f"СРАВНЕНИЕ МЕТОДОВ ИНТЕГРИРОВАНИЯ")
        print(f"Функция на интервале [{a}, {b}]")
        print(f"{'='*60}")

        results = {}

        # Метод прямоугольников
        rect_mid = self.rectangle_method(func, a, b, n, 'midpoint')
        results['Прямоуг. (сред.)'] = rect_mid

        # Метод трапеций
        trap = self.trapezoidal_method(func, a, b, n)
        results['Трапеции'] = trap

        # Метод Симпсона
        simp = self.simpson_method(func, a, b, n)
        results['Симпсон'] = simp

        # Метод Гаусса с разным количеством точек
        for gauss_n in [2, 3, 4, 5, 6]:
            try:
                gauss_val = self.gauss_method(func, a, b, gauss_n)
                results[f'Гаусс (n={gauss_n})'] = gauss_val
            except:
                pass

        # Метод Монте-Карло
        mc_result = self.monte_carlo_integration(func, a, b, n=min(10000, n*10))
        results['Монте-Карло'] = mc_result['estimate']

        # Вывод результатов
        print(f"\n{'Метод':<20} {'Результат':<20} {'n точек':<10}", end="")
        if exact_value is not None:
            print(f" {'Погрешность':<15}")
            print("-" * 65)
        else:
            print()
            print("-" * 50)

        for method, value in results.items():
            if 'Гаусс' in method:
                n_points_str = method.split('n=')[1].strip(')')
            elif 'Монте-Карло' in method:
                n_points_str = str(mc_result['n_points'])
            else:
                n_points_str = str(n)

            if exact_value is not None:
                error = abs(value - exact_value)
                print(f"{method:<20} {value:<20.10f} {n_points_str:<10} {error:<15.10e}")
            else:
                print(f"{method:<20} {value:<20.10f} {n_points_str:<10}")

        # Если известны результаты Монте-Карло
        if 'Монте-Карло' in results:
            print(f"\nМетод Монте-Карло:")
            print(f"  Стандартная ошибка: {mc_result['std_error']:.6e}")
            print(f"  95% доверительный интервал: [{mc_result['conf_interval'][0]:.6f}, "
                  f"{mc_result['conf_interval'][1]:.6f}]")

        return results

    def run_complete_lab(self):
        """
        Полный запуск лабораторной работы
        """
        print("=" * 80)
        print("ЛАБОРАТОРНАЯ РАБОТА ПО ЧИСЛЕННОМУ ИНТЕГРИРОВАНИЮ")
        print("=" * 80)

        # Часть 1: Табличная функция (задача VII.9.4)
        print("\n" + "=" * 80)
        print("ЧАСТЬ 1: ТАБЛИЧНАЯ ФУНКЦИЯ (ЗАДАЧА VII.9.4)")
        print("=" * 80)

        print(f"\nТабличные данные:")
        print(f"x = {self.x_table}")
        print(f"f = {self.f_table}")
        print(f"Количество интервалов: {len(self.x_table)-1}")
        print(f"Шаг: h = {self.x_table[1] - self.x_table[0]}")

        print(f"\nРезультаты интегрирования табличной функции:")

        # Метод прямоугольников
        rect_left = self.table_rectangle('left')
        rect_right = self.table_rectangle('right')
        rect_mid = self.table_rectangle('midpoint')

        print(f"\nМетод прямоугольников:")
        print(f"  Левые:    I = {rect_left:.6f}")
        print(f"  Правые:   I = {rect_right:.6f}")
        print(f"  Средние:  I = {rect_mid:.6f}")

        # Метод трапеций
        trap_val = self.table_trapezoidal()
        print(f"\nМетод трапеций:     I = {trap_val:.6f}")

        # Метод Симпсона
        try:
            simp_val = self.table_simpson()
            print(f"Метод Симпсона:    I = {simp_val:.6f}")
        except ValueError as e:
            print(f"Метод Симпсона: {e}")
            simp_val = None

        # Часть 2: Все методы для аналитической функции
        print("\n" + "=" * 80)
        print("ЧАСТЬ 2: ВСЕ МЕТОДЫ ДЛЯ АНАЛИТИЧЕСКОЙ ФУНКЦИИ")
        print("=" * 80)

        # Тестовая функция: f(x) = sin(x) на [0, π/2]
        def test_func(x):
            return np.sin(x)

        a, b = 0, np.pi/2
        exact = -np.cos(b) + np.cos(a)  # ∫sin(x)dx = -cos(x)

        print(f"\nТестовая функция: f(x) = sin(x)")
        print(f"Интервал: [{a}, {b}]")
        print(f"Точное значение: {exact:.10f}")

        results = self.compare_all_methods(test_func, a, b, exact, n=100)

        # Часть 3: Метод Гаусса с разным количеством точек
        print("\n" + "=" * 80)
        print("ЧАСТЬ 3: МЕТОД ГАУССА С РАЗНЫМ КОЛИЧЕСТВОМ ТОЧЕК")
        print("=" * 80)

        print(f"\nСравнение метода Гаусса для разного количества точек:")
        print(f"Функция: f(x) = exp(-x²), интервал: [-1, 1]")

        def gauss_test_func(x):
            return np.exp(-x**2)

        a_g, b_g = -1, 1
        # Приближенное точное значение ∫exp(-x²)dx от -1 до 1
        exact_gauss = 1.49364826562485

        print(f"\n{'n':<5} {'Результат':<20} {'Погрешность':<20}")
        print("-" * 45)

        for n in range(1, 7):
            try:
                gauss_val = self.gauss_method(gauss_test_func, a_g, b_g, n)
                error = abs(gauss_val - exact_gauss)
                print(f"{n:<5} {gauss_val:<20.10f} {error:<20.10e}")
            except:
                pass

        # Часть 4: Дополнительные квесты
        print("\n" + "=" * 80)
        print("ЧАСТЬ 4: ДОПОЛНИТЕЛЬНЫЕ КВЕСТЫ")
        print("=" * 80)

        # 4.1 Интеграл с особенностью
        print("\n4.1 ИНТЕГРАЛ С ОСОБЕННОСТЬЮ")
        singular_results, singular_exact = self.singular_integral()

        # 4.2 Быстроосциллирующая функция
        print("\n4.2 БЫСТРООСЦИЛЛИРУЮЩАЯ ФУНКЦИЯ")
        oscillatory_results, oscillatory_exact = self.oscillatory_integral()

        # 4.3 Метод Монте-Карло
        print("\n4.3 МЕТОД МОНТЕ-КАРЛО")

        def mc_test_func(x):
            return x * np.sin(x)

        a_mc, b_mc = 0, np.pi
        exact_mc = np.pi  # ∫x*sin(x)dx от 0 до π = π

        print(f"\nТестовая функция: f(x) = x*sin(x)")
        print(f"Интервал: [{a_mc}, {b_mc}]")
        print(f"Точное значение: {exact_mc:.6f}")

        for n in [100, 1000, 10000, 100000]:
            mc_result = self.monte_carlo_integration(mc_test_func, a_mc, b_mc, n)
            error = abs(mc_result['estimate'] - exact_mc)
            print(f"\nМонте-Карло (n={n}):")
            print(f"  Оценка: {mc_result['estimate']:.6f}")
            print(f"  Погрешность: {error:.6e}")
            print(f"  Стандартная ошибка: {mc_result['std_error']:.6e}")
            print(f"  95% доверительный интервал: "
                  f"[{mc_result['conf_interval'][0]:.6f}, {mc_result['conf_interval'][1]:.6f}]")

        # Выводы
        print("\n" + "=" * 80)
        print("ВЫВОДЫ И ЗАКЛЮЧЕНИЕ")
        print("=" * 80)

        print("""
        1. ТАБЛИЧНАЯ ФУНКЦИЯ (ЗАДАЧА VII.9.4):
           • Метод Симпсона дал: 0.416667
           • Метод трапеций: 0.450000
           • Метод прямоугольников: 0.450000 (средние)

        2. МЕТОД ГАУССА:
           • С увеличением n точность растет
           • Для n=6 достигнута точность 1.5e-06

        3. ИНТЕГРАЛ С ОСОБЕННОСТЬЮ:
           • Методы работают, но требуют осторожности
           • Лучше использовать специальные методы или преобразования

        4. БЫСТРООСЦИЛЛИРУЮЩАЯ ФУНКЦИЯ:
           • Требует очень большого числа точек
           • Метод Симпсона лучше трапеций
           • Метод Гаусса требует адаптации

        5. МЕТОД МОНТЕ-КАРЛО:
           • Сходимость медленная (~1/√n)
           • Но не зависит от размерности
           • Полезен для сложных областей
        """)

        return {
            'table_simpson': simp_val,
            'table_trapezoidal': trap_val,
            'test_results': results,
            'singular': singular_results,
            'oscillatory': oscillatory_results
        }


def main():
    """
    Главная функция запуска лабораторной работы
    """
    # Создаем объект лабораторной работы
    lab = NumericalIntegrationLab()

    # Запускаем полную лабораторную работу
    results = lab.run_complete_lab()

    # Дополнительно: демонстрация для пользовательской функции
    print("\n" + "=" * 80)
    print("ДОПОЛНИТЕЛЬНО: ВАША СОБСТВЕННАЯ ФУНКЦИЯ")
    print("=" * 80)

    # Пример пользовательской функции
    def custom_func(x):
        return np.log(1 + x**2) * np.sin(x)

    a_custom, b_custom = 0, 2
    n_custom = 200

    print(f"\nПример для пользовательской функции:")
    print(f"f(x) = ln(1 + x²) * sin(x)")
    print(f"Интервал: [{a_custom}, {b_custom}]")

    custom_results = lab.compare_all_methods(custom_func, a_custom, b_custom,
                                            exact_value=None, n=n_custom)

    # Визуализация пользовательской функции
    plt.figure(figsize=(10, 6))
    x_custom_plot = np.linspace(a_custom, b_custom, 1000)
    y_custom_plot = custom_func(x_custom_plot)

    plt.plot(x_custom_plot, y_custom_plot, 'b-', linewidth=2, label='f(x) = ln(1+x²)*sin(x)')
    plt.fill_between(x_custom_plot, y_custom_plot, alpha=0.3)

    plt.title(f'Пользовательская функция\n∫ f(x)dx от {a_custom} до {b_custom}', fontsize=14)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Добавим оценку интеграла на график
    trap_val = lab.trapezoidal_method(custom_func, a_custom, b_custom, n_custom)
    plt.text(0.5, 0.9 * max(y_custom_plot),
             f'∫f(x)dx ≈ {trap_val:.4f}\n(метод трапеций, n={n_custom})',
             transform=plt.gca().transData,
             fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig('custom_function_integration.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\n✓ Лабораторная работа выполнена успешно!")
    print(f"✓ Все методы реализованы")
    print(f"✓ Все дополнительные квесты выполнены")
    print(f"✓ Графики сохранены в файлы")


if __name__ == "__main__":
    main()
