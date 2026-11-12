"""Lazy field access and whole-script cleanup without hardware."""
import contextlib
import io
import unittest
from unittest.mock import patch

from vsss import Car, GPS, gps, run


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.radio_patch = patch('vsss.car.MecanumBLEClient')
        self.radio = self.radio_patch.start()
        self.addCleanup(self.radio_patch.stop)

    def tearDown(self):
        # Also recover cars whose cleanup was deliberately failed by a test.
        with Car._cars_lock:
            Car._cars.clear()

    def test_velocity_only_never_constructs_gps_and_cleanup_is_idempotent(self):
        with patch('vsss.car.GPS', side_effect=AssertionError('GPS requested')):
            car = Car(1)
            car.set_velocity(x=.2)
            Car.close_all()
            Car.close_all()
            car.close()
        self.radio.return_value.disconnect.assert_called_once()

    def test_car_field_access_is_lazy_and_uses_same_singleton(self):
        with patch('vsss.car.GPS') as factory:
            car = Car(1)
            factory.assert_not_called()
            self.assertIs(car.gps, factory.return_value)
            factory.assert_called_once()
            car.close()

    def test_global_field_and_close_do_not_start_receiver_early(self):
        self.assertIsNone(GPS._instance)
        gps.close()
        self.assertIsNone(GPS._instance)
        with patch('vsss.gps.GPS') as factory:
            factory.return_value.cars = {1: 'pose'}
            self.assertEqual(gps.cars, {1: 'pose'})
            factory.assert_called_once()
            self.radio.assert_not_called()

    def test_run_cleans_up_first_car_when_second_connection_fails(self):
        @run
        def main():
            Car(1)
            self.radio.return_value.wait_until_connected.side_effect = TimeoutError('car2 missing')
            Car(2)

        with self.assertRaisesRegex(TimeoutError, 'car2 missing'):
            main()
        self.assertFalse(Car._cars)
        self.assertEqual(self.radio.return_value.disconnect.call_count, 2)

    def test_close_all_attempts_every_car_and_retains_failed_one_for_retry(self):
        first, second = Car(1), Car(2)
        with patch.object(first, 'close', side_effect=RuntimeError('disconnect failed')), \
                patch.object(second, 'close', wraps=second.close) as close_second:
            with self.assertRaisesRegex(RuntimeError, 'disconnect failed'):
                Car.close_all()
            close_second.assert_called_once()
        self.assertIn(first, Car._cars)
        Car.close_all()
        self.assertFalse(Car._cars)

    def test_run_retries_cleanup_of_partially_constructed_car(self):
        self.radio.return_value.wait_until_connected.side_effect = TimeoutError('connection failed')
        self.radio.return_value.disconnect.side_effect = [RuntimeError('temporary close failure'), None]

        @run
        def main():
            Car(1)

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(TimeoutError, 'connection failed'):
                main()
        self.assertEqual(self.radio.return_value.disconnect.call_count, 2)
        self.assertFalse(Car._cars)

    def test_run_preserves_return_metadata_and_closes_all_cars(self):
        @run
        def main(value):
            """Example entry point."""
            Car(1)
            Car(2)
            return value

        with patch.object(gps, 'close') as close_gps:
            self.assertEqual(main(42), 42)
            self.assertFalse(Car._cars)
            close_gps.assert_called_once()
        self.assertEqual(self.radio.return_value.disconnect.call_count, 2)
        self.assertEqual(main.__name__, 'main')
        self.assertEqual(main.__doc__, 'Example entry point.')

    def test_run_preserves_script_error_and_ctrl_c_when_cleanup_also_fails(self):
        for error in (ValueError('script failed'), KeyboardInterrupt()):
            @run
            def main():
                raise error

            with patch.object(Car, 'close_all', side_effect=RuntimeError('cleanup failed')), \
                    patch.object(gps, 'close') as close_gps, \
                    contextlib.redirect_stderr(io.StringIO()) as output:
                with self.assertRaises(type(error)) as caught:
                    main()
                self.assertIs(caught.exception, error)
                close_gps.assert_called_once()
                self.assertIn('cleanup failed', output.getvalue())

    def test_cleanup_errors_propagate_after_success_even_inside_outer_except(self):
        @run
        def main():
            return 1

        with patch.object(Car, 'close_all', side_effect=RuntimeError('cleanup failed')), \
                patch.object(gps, 'close') as close_gps:
            try:
                raise ValueError('outer')
            except ValueError:
                with self.assertRaisesRegex(RuntimeError, 'cleanup failed'):
                    main()
            close_gps.assert_called_once()

    def test_async_and_generator_functions_are_rejected(self):
        async def asynchronous():
            pass

        def generator():
            yield 1

        for function in (asynchronous, generator):
            with self.assertRaises(TypeError):
                run(function)


if __name__ == '__main__':
    unittest.main()
