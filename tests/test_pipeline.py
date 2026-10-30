from contextlib import closing
from unittest import mock

import pytest
from redis import RedisClusterException
import redis
from redis._parsers.encoders import Encoder
from redis.client import Pipeline
from redis.commands.core import Script
from redis.observability import recorder
from redis.observability.config import OTelConfig, MetricGroup
from redis.observability.metrics import RedisMetricsCollector
from redis.pipeline_policy import (
    EffectivePipelinePolicy,
    ExecMode,
    ScriptPreload,
    StackEmptyMode,
    WatchDispatch,
    WatchFault,
)

from .conftest import skip_if_server_version_lt, wait_for_command


class TestPipeline:
    def test_pipeline_is_true(self, r):
        "Ensure pipeline instances are not false-y"
        with r.pipeline() as pipe:
            assert pipe

    def test_pipeline(self, r):
        with r.pipeline() as pipe:
            (
                pipe.set("a", "a1")
                .get("a")
                .zadd("z", {"z1": 1})
                .zadd("z", {"z2": 4})
                .zincrby("z", 1, "z1")
            )
            assert pipe.execute() == [
                True,
                b"a1",
                True,
                True,
                2.0,
            ]

    def test_pipeline_memoryview(self, r):
        with r.pipeline() as pipe:
            (pipe.set("a", memoryview(b"a1")).get("a"))
            assert pipe.execute() == [True, b"a1"]

    def test_pipeline_length(self, r):
        with r.pipeline() as pipe:
            # Initially empty.
            assert len(pipe) == 0

            # Fill 'er up!
            pipe.set("a", "a1").set("b", "b1").set("c", "c1")
            assert len(pipe) == 3

            # Execute calls reset(), so empty once again.
            pipe.execute()
            assert len(pipe) == 0

    def test_pipeline_no_transaction(self, r):
        with r.pipeline(transaction=False) as pipe:
            pipe.set("a", "a1").set("b", "b1").set("c", "c1")
            assert pipe.execute() == [True, True, True]
            assert r["a"] == b"a1"
            assert r["b"] == b"b1"
            assert r["c"] == b"c1"

    @pytest.mark.onlynoncluster
    def test_pipeline_no_transaction_watch(self, r):
        r["a"] = 0

        with r.pipeline(transaction=False) as pipe:
            pipe.watch("a")
            a = pipe.get("a")

            pipe.multi()
            pipe.set("a", int(a) + 1)
            assert pipe.execute() == [True]

    @pytest.mark.onlynoncluster
    def test_pipeline_no_transaction_watch_failure(self, r):
        r["a"] = 0

        with r.pipeline(transaction=False) as pipe:
            pipe.watch("a")
            a = pipe.get("a")

            r["a"] = "bad"

            pipe.multi()
            pipe.set("a", int(a) + 1)

            with pytest.raises(redis.WatchError):
                pipe.execute()

            assert r["a"] == b"bad"

    def test_exec_error_in_response(self, r):
        """
        an invalid pipeline command at exec time adds the exception instance
        to the list of returned values
        """
        r["c"] = "a"
        with r.pipeline() as pipe:
            pipe.set("a", 1).set("b", 2).lpush("c", 3).set("d", 4)
            result = pipe.execute(raise_on_error=False)

            assert result[0]
            assert r["a"] == b"1"
            assert result[1]
            assert r["b"] == b"2"

            # we can't lpush to a key that's a string value, so this should
            # be a ResponseError exception
            assert isinstance(result[2], redis.ResponseError)
            assert r["c"] == b"a"

            # since this isn't a transaction, the other commands after the
            # error are still executed
            assert result[3]
            assert r["d"] == b"4"

            # make sure the pipe was restored to a working state
            assert pipe.set("z", "zzz").execute() == [True]
            assert r["z"] == b"zzz"

    def test_exec_error_raised(self, r):
        r["c"] = "a"
        with r.pipeline() as pipe:
            pipe.set("a", 1).set("b", 2).lpush("c", 3).set("d", 4)
            with pytest.raises(redis.ResponseError) as ex:
                pipe.execute()
            assert str(ex.value).startswith(
                "Command # 3 (LPUSH c 3) of pipeline caused error: "
            )

            # make sure the pipe was restored to a working state
            assert pipe.set("z", "zzz").execute() == [True]
            assert r["z"] == b"zzz"

    @pytest.mark.onlynoncluster
    def test_transaction_with_empty_error_command(self, r):
        """
        Commands with custom EMPTY_ERROR functionality return their default
        values in the pipeline no matter the raise_on_error preference
        """
        for error_switch in (True, False):
            with r.pipeline() as pipe:
                pipe.set("a", 1).mget([]).set("c", 3)
                result = pipe.execute(raise_on_error=error_switch)

                assert result[0]
                assert result[1] == []
                assert result[2]

    @pytest.mark.onlynoncluster
    def test_pipeline_with_empty_error_command(self, r):
        """
        Commands with custom EMPTY_ERROR functionality return their default
        values in the pipeline no matter the raise_on_error preference
        """
        for error_switch in (True, False):
            with r.pipeline(transaction=False) as pipe:
                pipe.set("a", 1).mget([]).set("c", 3)
                result = pipe.execute(raise_on_error=error_switch)

                assert result[0]
                assert result[1] == []
                assert result[2]

    def test_parse_error_raised(self, r):
        with r.pipeline() as pipe:
            # the zrem is invalid because we don't pass any keys to it
            pipe.set("a", 1).zrem("b").set("b", 2)
            with pytest.raises(redis.ResponseError) as ex:
                pipe.execute()

            assert str(ex.value).startswith(
                "Command # 2 (ZREM b) of pipeline caused error: "
            )

            # make sure the pipe was restored to a working state
            assert pipe.set("z", "zzz").execute() == [True]
            assert r["z"] == b"zzz"

    @pytest.mark.onlynoncluster
    def test_parse_error_raised_transaction(self, r):
        with r.pipeline() as pipe:
            pipe.multi()
            # the zrem is invalid because we don't pass any keys to it
            pipe.set("a", 1).zrem("b").set("b", 2)
            with pytest.raises(redis.ResponseError) as ex:
                pipe.execute()

            assert str(ex.value).startswith(
                "Command # 2 (ZREM b) of pipeline caused error: "
            )

            # make sure the pipe was restored to a working state
            assert pipe.set("z", "zzz").execute() == [True]
            assert r["z"] == b"zzz"

    @pytest.mark.onlynoncluster
    def test_watch_succeed(self, r):
        r["a"] = 1
        r["b"] = 2

        with r.pipeline() as pipe:
            pipe.watch("a", "b")
            assert pipe.watching
            a_value = pipe.get("a")
            b_value = pipe.get("b")
            assert a_value == b"1"
            assert b_value == b"2"
            pipe.multi()

            pipe.set("c", 3)
            assert pipe.execute() == [True]
            assert not pipe.watching

    @pytest.mark.onlynoncluster
    def test_watch_failure(self, r):
        r["a"] = 1
        r["b"] = 2

        with r.pipeline() as pipe:
            pipe.watch("a", "b")
            r["b"] = 3
            pipe.multi()
            pipe.get("a")
            with pytest.raises(redis.WatchError):
                pipe.execute()

            assert not pipe.watching

    @pytest.mark.onlynoncluster
    def test_watch_failure_in_empty_transaction(self, r):
        r["a"] = 1
        r["b"] = 2

        with r.pipeline() as pipe:
            pipe.watch("a", "b")
            r["b"] = 3
            pipe.multi()
            with pytest.raises(redis.WatchError):
                pipe.execute()

            assert not pipe.watching

    @pytest.mark.onlynoncluster
    def test_unwatch(self, r):
        r["a"] = 1
        r["b"] = 2

        with r.pipeline() as pipe:
            pipe.watch("a", "b")
            r["b"] = 3
            pipe.unwatch()
            assert not pipe.watching
            pipe.get("a")
            assert pipe.execute() == [b"1"]

    @pytest.mark.onlynoncluster
    def test_watch_exec_no_unwatch(self, r):
        r["a"] = 1
        r["b"] = 2

        with r.monitor() as m:
            with r.pipeline() as pipe:
                pipe.watch("a", "b")
                assert pipe.watching
                a_value = pipe.get("a")
                b_value = pipe.get("b")
                assert a_value == b"1"
                assert b_value == b"2"
                pipe.multi()
                pipe.set("c", 3)
                assert pipe.execute() == [True]
                assert not pipe.watching

            unwatch_command = wait_for_command(r, m, "UNWATCH")
            assert unwatch_command is None, "should not send UNWATCH"

    @pytest.mark.onlynoncluster
    def test_watch_reset_unwatch(self, r):
        r["a"] = 1

        with r.monitor() as m:
            with r.pipeline() as pipe:
                pipe.watch("a")
                assert pipe.watching
                pipe.reset()
                assert not pipe.watching

            unwatch_command = wait_for_command(r, m, "UNWATCH")
            assert unwatch_command is not None
            assert unwatch_command["command"] == "UNWATCH"

    @pytest.mark.onlynoncluster
    def test_close_is_reset(self, r):
        with r.pipeline() as pipe:
            called = 0

            def mock_reset():
                nonlocal called
                called += 1

            with mock.patch.object(pipe, "reset", mock_reset):
                pipe.close()
                assert called == 1

    @pytest.mark.onlynoncluster
    def test_closing(self, r):
        with closing(r.pipeline()):
            pass

    @pytest.mark.onlynoncluster
    def test_transaction_callable(self, r):
        r["a"] = 1
        r["b"] = 2
        has_run = []

        def my_transaction(pipe):
            a_value = pipe.get("a")
            assert a_value in (b"1", b"2")
            b_value = pipe.get("b")
            assert b_value == b"2"

            # silly run-once code... incr's "a" so WatchError should be raised
            # forcing this all to run again. this should incr "a" once to "2"
            if not has_run:
                r.incr("a")
                has_run.append("it has")

            pipe.multi()
            pipe.set("c", int(a_value) + int(b_value))

        result = r.transaction(my_transaction, "a", "b")
        assert result == [True]
        assert r["c"] == b"4"

    @pytest.mark.onlynoncluster
    def test_transaction_callable_returns_value_from_callable(self, r):
        def callback(pipe):
            # No need to do anything here since we only want the return value
            return "a"

        res = r.transaction(callback, "my-key", value_from_callable=True)
        assert res == "a"

    def test_exec_error_in_no_transaction_pipeline(self, r):
        r["a"] = 1
        with r.pipeline(transaction=False) as pipe:
            pipe.llen("a")
            pipe.expire("a", 100)

            with pytest.raises(redis.ResponseError) as ex:
                pipe.execute()

            assert str(ex.value).startswith(
                "Command # 1 (LLEN a) of pipeline caused error: "
            )

        assert r["a"] == b"1"

    def test_exec_error_in_no_transaction_pipeline_unicode_command(self, r):
        key = chr(3456) + "abcd" + chr(3421)
        r[key] = 1
        with r.pipeline(transaction=False) as pipe:
            pipe.llen(key)
            pipe.expire(key, 100)

            with pytest.raises(redis.ResponseError) as ex:
                pipe.execute()

            expected = f"Command # 1 (LLEN {key}) of pipeline caused error: "
            assert str(ex.value).startswith(expected)

        assert r[key] == b"1"

    def test_exec_error_in_pipeline_truncated(self, r):
        key = "a" * 50
        a_value = "a" * 20
        b_value = "b" * 20

        r[key] = 1
        with r.pipeline(transaction=False) as pipe:
            pipe.hset(key, mapping={"field_a": a_value, "field_b": b_value})
            pipe.expire(key, 100)

            with pytest.raises(redis.ResponseError) as ex:
                pipe.execute()

            expected = f"Command # 1 (HSET {key} field_a {a_value} field_b...) of pipeline caused error: "
            assert str(ex.value).startswith(expected)

    def test_pipeline_with_bitfield(self, r):
        with r.pipeline() as pipe:
            pipe.set("a", "1")
            bf = pipe.bitfield("b")
            pipe2 = (
                bf.set("u8", 8, 255)
                .get("u8", 0)
                .get("u4", 8)  # 1111
                .get("u4", 12)  # 1111
                .get("u4", 13)  # 1110
                .execute()
            )
            pipe.get("a")
            response = pipe.execute()

            assert pipe == pipe2
            assert response == [True, [0, 0, 15, 15, 14], b"1"]

    @pytest.mark.onlynoncluster
    @skip_if_server_version_lt("2.0.0")
    def test_pipeline_discard(self, r):
        # empty pipeline should raise an error
        with r.pipeline() as pipe:
            pipe.set("key", "someval")
            pipe.discard()
            with pytest.raises(redis.exceptions.ResponseError):
                pipe.execute()

        # setting a pipeline and discarding should do the same
        with r.pipeline() as pipe:
            pipe.set("key", "someval")
            pipe.set("someotherkey", "val")
            response = pipe.execute()
            pipe.set("key", "another value!")
            pipe.discard()
            pipe.set("key", "another vae!")
            with pytest.raises(redis.exceptions.ResponseError):
                pipe.execute()

            pipe.set("foo", "bar")
            response = pipe.execute()
        assert response[0]
        assert r.get("foo") == b"bar"

    @pytest.mark.onlynoncluster
    def test_send_set_commands_over_pipeline(self, r: redis.Redis):
        pipe = r.pipeline()
        pipe.hset("hash:1", "foo", "bar")
        pipe.hset("hash:1", "bar", "foo")
        pipe.hset("hash:1", "baz", "bar")
        pipe.hgetall("hash:1")
        resp = pipe.execute()
        assert resp == [1, 1, 1, {b"bar": b"foo", b"baz": b"bar", b"foo": b"bar"}]

    @pytest.mark.onlycluster
    @skip_if_server_version_lt("8.3.224")
    def test_pipeline_with_msetex(self, r):
        r.delete("key1", "key2", "key1_transaction", "key2_transaction")

        p = r.pipeline()
        with pytest.raises(RedisClusterException):
            p.msetex({"key1": "value1", "key2": "value2"}, ex=1000)

        p_transaction = r.pipeline(transaction=True)
        with pytest.raises(RedisClusterException):
            p_transaction.msetex(
                {"key1_transaction": "value1", "key2_transaction": "value2"}, ex=10
            )


class TestPipelineMetricsRecording:
    """
    Unit tests that verify metrics are properly recorded from Pipeline
    and delivered to the Meter through the observability recorder.

    These tests use fully mocked connection and connection pool - no real Redis
    or OTel integration is used.
    """

    @pytest.fixture
    def mock_connection(self):
        """Create a mock connection with required attributes."""
        conn = mock.MagicMock()
        conn.host = "localhost"
        conn.port = 6379
        conn.db = 0

        # Mock retry to just execute the function directly
        def mock_call_with_retry(do, fail, is_retryable=None, with_failure_count=False):
            return do()

        conn.retry.call_with_retry = mock_call_with_retry

        return conn

    @pytest.fixture
    def mock_connection_pool(self, mock_connection):
        """Create a mock connection pool."""
        pool = mock.MagicMock()
        pool.get_connection.return_value = mock_connection
        pool.get_encoder.return_value = mock.MagicMock()
        return pool

    @pytest.fixture
    def mock_meter(self):
        """Create a mock Meter that tracks all instrument calls."""
        meter = mock.MagicMock()

        # Create mock histogram for operation duration
        self.operation_duration = mock.MagicMock()
        # Create mock counter for client errors
        self.client_errors = mock.MagicMock()

        def create_histogram_side_effect(name, **kwargs):
            if name == "db.client.operation.duration":
                return self.operation_duration
            return mock.MagicMock()

        def create_counter_side_effect(name, **kwargs):
            if name == "redis.client.errors":
                return self.client_errors
            return mock.MagicMock()

        meter.create_counter.side_effect = create_counter_side_effect
        meter.create_up_down_counter.return_value = mock.MagicMock()
        meter.create_histogram.side_effect = create_histogram_side_effect

        return meter

    @pytest.fixture
    def setup_pipeline_with_otel(
        self, mock_connection_pool, mock_connection, mock_meter
    ):
        """
        Setup a Pipeline with mocked connection and OTel collector.
        Returns tuple of (pipeline, operation_duration_mock).
        """

        # Reset any existing collector state
        recorder.reset_collector()

        # Create config with COMMAND group enabled
        config = OTelConfig(metric_groups=[MetricGroup.COMMAND])

        # Create collector with mocked meter
        with mock.patch("redis.observability.metrics.OTEL_AVAILABLE", True):
            collector = RedisMetricsCollector(mock_meter, config)

        # Patch the recorder to use our collector
        with mock.patch.object(
            recorder, "_get_or_create_collector", return_value=collector
        ):
            # Create pipeline with mocked connection pool
            pipeline = Pipeline(
                connection_pool=mock_connection_pool,
                response_callbacks={},
                transaction=True,
                shard_hint=None,
            )

            yield pipeline, self.operation_duration

        # Cleanup
        recorder.reset_collector()

    def test_pipeline_execute_records_metric(self, setup_pipeline_with_otel):
        """
        Test that executing a pipeline records operation duration metric
        which is delivered to the Meter's histogram.record() method.
        """
        pipeline, operation_duration_mock = setup_pipeline_with_otel

        # Mock _execute_transaction to return successful responses
        pipeline._execute_transaction = mock.MagicMock(
            return_value=[True, True, b"value1"]
        )

        # Queue commands in the pipeline
        pipeline.command_stack = [
            (("SET", "key1", "value1"), {}),
            (("SET", "key2", "value2"), {}),
            (("GET", "key1"), {}),
        ]

        # Execute the pipeline
        pipeline.execute()

        # Verify the Meter's histogram.record() was called
        operation_duration_mock.record.assert_called_once()

        # Get the call arguments
        call_args = operation_duration_mock.record.call_args

        # Verify duration was recorded (first positional arg)
        duration = call_args[0][0]
        assert isinstance(duration, float)
        assert duration >= 0

        # Verify attributes
        attrs = call_args[1]["attributes"]
        assert attrs["db.operation.name"] == "MULTI"
        # Note: db.operation.batch.size is no longer included in OTel attributes
        assert "db.operation.batch.size" not in attrs
        assert attrs["server.address"] == "localhost"
        assert attrs["server.port"] == 6379
        assert attrs["db.namespace"] == "0"

    def test_pipeline_no_transaction_records_pipeline_command_name(
        self, mock_connection_pool, mock_connection, mock_meter
    ):
        """
        Test that executing a pipeline without transaction
        records metric with command_name='PIPELINE'.
        """
        recorder.reset_collector()
        config = OTelConfig(metric_groups=[MetricGroup.COMMAND])

        with mock.patch("redis.observability.metrics.OTEL_AVAILABLE", True):
            collector = RedisMetricsCollector(mock_meter, config)

        with mock.patch.object(
            recorder, "_get_or_create_collector", return_value=collector
        ):
            # Create pipeline with transaction=False
            pipeline = Pipeline(
                connection_pool=mock_connection_pool,
                response_callbacks={},
                transaction=False,  # Non-transaction mode
                shard_hint=None,
            )

            pipeline._execute_pipeline = mock.MagicMock(return_value=[True, True])
            pipeline.command_stack = [
                (("SET", "key1", "value1"), {}),
                (("SET", "key2", "value2"), {}),
            ]

            pipeline.execute()

            # Verify command name is PIPELINE
            call_args = self.operation_duration.record.call_args
            attrs = call_args[1]["attributes"]
            assert attrs["db.operation.name"] == "PIPELINE"

        recorder.reset_collector()

    def test_pipeline_error_records_error_count(
        self, mock_connection_pool, mock_connection, mock_meter
    ):
        """
        Test that when a pipeline execution raises an exception,
        record_error_count is called (not record_operation_duration).
        Direct failures call record_error_count, not record_operation_duration.
        """
        recorder.reset_collector()
        # Enable RESILIENCY metric group for error counting
        config = OTelConfig(metric_groups=[MetricGroup.COMMAND, MetricGroup.RESILIENCY])

        with mock.patch("redis.observability.metrics.OTEL_AVAILABLE", True):
            collector = RedisMetricsCollector(mock_meter, config)

        with mock.patch.object(
            recorder, "_get_or_create_collector", return_value=collector
        ):
            pipeline = Pipeline(
                connection_pool=mock_connection_pool,
                response_callbacks={},
                transaction=False,
                shard_hint=None,
            )

            # Make execute raise an exception
            test_error = redis.ResponseError("WRONGTYPE Operation error")
            pipeline._execute_pipeline = mock.MagicMock(side_effect=test_error)
            pipeline.command_stack = [(("LPUSH", "string_key", "value"), {})]

            # Execute should raise the error
            with pytest.raises(redis.ResponseError):
                pipeline.execute()

            # For direct failures (no retries), record_error_count is called
            # record_operation_duration is NOT called for direct failures
            self.operation_duration.record.assert_not_called()

            # Verify record_error_count was called
            self.client_errors.add.assert_called_once()
            error_call_args = self.client_errors.add.call_args
            error_attrs = error_call_args[1]["attributes"]
            assert "error.type" in error_attrs

        recorder.reset_collector()

    def test_pipeline_server_attributes_recorded(self, setup_pipeline_with_otel):
        """
        Test that server address, port, and db namespace are correctly recorded.
        """
        pipeline, operation_duration_mock = setup_pipeline_with_otel

        pipeline._execute_transaction = mock.MagicMock(return_value=[True])
        pipeline.command_stack = [(("PING",), {})]

        pipeline.execute()

        call_args = operation_duration_mock.record.call_args
        attrs = call_args[1]["attributes"]

        # Verify server attributes match mock connection
        assert attrs["server.address"] == "localhost"
        assert attrs["server.port"] == 6379
        assert attrs["db.namespace"] == "0"

    def test_multiple_pipeline_executions_record_multiple_metrics(
        self, mock_connection_pool, mock_connection, mock_meter
    ):
        """
        Test that each pipeline execution records a separate metric to the Meter.
        """
        recorder.reset_collector()
        config = OTelConfig(metric_groups=[MetricGroup.COMMAND])

        with mock.patch("redis.observability.metrics.OTEL_AVAILABLE", True):
            collector = RedisMetricsCollector(mock_meter, config)

        with mock.patch.object(
            recorder, "_get_or_create_collector", return_value=collector
        ):
            # First pipeline execution
            pipeline1 = Pipeline(
                connection_pool=mock_connection_pool,
                response_callbacks={},
                transaction=True,
                shard_hint=None,
            )
            pipeline1._execute_transaction = mock.MagicMock(return_value=[True])
            pipeline1.command_stack = [(("SET", "key1", "value1"), {})]
            pipeline1.execute()

            # Second pipeline execution
            pipeline2 = Pipeline(
                connection_pool=mock_connection_pool,
                response_callbacks={},
                transaction=True,
                shard_hint=None,
            )
            pipeline2._execute_transaction = mock.MagicMock(return_value=[True, True])
            pipeline2.command_stack = [
                (("SET", "key2", "value2"), {}),
                (("SET", "key3", "value3"), {}),
            ]
            pipeline2.execute()

            # Verify histogram.record() was called twice
            assert self.operation_duration.record.call_count == 2

        recorder.reset_collector()

    def test_empty_pipeline_does_not_record_metric(
        self, mock_connection_pool, mock_connection, mock_meter
    ):
        """
        Test that an empty pipeline (no commands) does not record a metric.
        """
        recorder.reset_collector()
        config = OTelConfig(metric_groups=[MetricGroup.COMMAND])

        with mock.patch("redis.observability.metrics.OTEL_AVAILABLE", True):
            collector = RedisMetricsCollector(mock_meter, config)

        with mock.patch.object(
            recorder, "_get_or_create_collector", return_value=collector
        ):
            pipeline = Pipeline(
                connection_pool=mock_connection_pool,
                response_callbacks={},
                transaction=True,
                shard_hint=None,
            )

            # Empty command stack
            pipeline.command_stack = []

            # Execute empty pipeline
            result = pipeline.execute()

            # Should return empty list
            assert result == []

            # No metric should be recorded for empty pipeline
            self.operation_duration.record.assert_not_called()

        recorder.reset_collector()

    def test_pipeline_retry_records_metric_on_each_attempt(
        self, mock_connection_pool, mock_meter
    ):
        """
        Test that when a pipeline is retried, operation duration metric
        is recorded for each retry attempt with retry_attempts attribute.
        """
        # Create connection with retry behavior
        mock_connection = mock.MagicMock()
        mock_connection.host = "localhost"
        mock_connection.port = 6379
        mock_connection.db = 0

        # Track retry attempts
        max_retries = 2

        def call_with_retry_impl(do, fail, is_retryable=None, with_failure_count=False):
            """Simulate retry behavior - fail twice, then succeed."""
            for attempt in range(max_retries + 1):
                try:
                    return do()
                except redis.ConnectionError as e:
                    if attempt < max_retries:
                        if with_failure_count:
                            fail(e, attempt + 1)
                        else:
                            fail(e)
                    else:
                        raise

        mock_connection.retry.call_with_retry = call_with_retry_impl
        mock_connection.retry.get_retries.return_value = max_retries

        mock_connection_pool.get_connection.return_value = mock_connection

        recorder.reset_collector()
        config = OTelConfig(metric_groups=[MetricGroup.COMMAND])

        with mock.patch("redis.observability.metrics.OTEL_AVAILABLE", True):
            collector = RedisMetricsCollector(mock_meter, config)

        with mock.patch.object(
            recorder, "_get_or_create_collector", return_value=collector
        ):
            pipeline = Pipeline(
                connection_pool=mock_connection_pool,
                response_callbacks={},
                transaction=False,
                shard_hint=None,
            )

            # Make pipeline fail twice then succeed
            call_count = [0]

            def execute_pipeline_impl(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] <= 2:
                    raise redis.ConnectionError("Connection failed")
                return [True, True]

            pipeline._execute_pipeline = mock.MagicMock(
                side_effect=execute_pipeline_impl
            )
            pipeline.command_stack = [
                (("SET", "key1", "value1"), {}),
                (("SET", "key2", "value2"), {}),
            ]

            # Execute pipeline - should retry twice then succeed
            pipeline.execute()

            # Verify histogram.record() was called 3 times:
            # 2 retry attempts + 1 final success
            assert self.operation_duration.record.call_count == 3

            calls = self.operation_duration.record.call_args_list

            # First two calls should have error.type (retry attempts)
            assert "error.type" in calls[0][1]["attributes"]
            assert "error.type" in calls[1][1]["attributes"]

            # Last call should be success (no error.type)
            assert "error.type" not in calls[2][1]["attributes"]

            # Note: db.operation.batch.size is no longer included in OTel attributes
            for call in calls:
                assert "db.operation.batch.size" not in call[1]["attributes"]

        recorder.reset_collector()

    def test_pipeline_retry_exhausted_records_final_error_metrics(
        self, mock_connection_pool, mock_meter
    ):
        """
        Test that when all retries are exhausted, metrics are recorded correctly:
        - record_operation_duration is called for each retry attempt (with error info)
        - record_error_count is called for the final error (after all retries exhausted)
        """
        mock_connection = mock.MagicMock()
        mock_connection.host = "localhost"
        mock_connection.port = 6379
        mock_connection.db = 0

        max_retries = 2

        def call_with_retry_impl(do, fail, is_retryable=None, with_failure_count=False):
            """Simulate retry behavior - always fail."""
            for attempt in range(max_retries + 1):
                try:
                    return do()
                except redis.ConnectionError as e:
                    if attempt < max_retries:
                        if with_failure_count:
                            fail(e, attempt + 1)
                        else:
                            fail(e)
                    else:
                        raise

        mock_connection.retry.call_with_retry = call_with_retry_impl
        mock_connection.retry.get_retries.return_value = max_retries

        mock_connection_pool.get_connection.return_value = mock_connection

        recorder.reset_collector()
        # Enable both COMMAND and RESILIENCY metric groups
        config = OTelConfig(metric_groups=[MetricGroup.COMMAND, MetricGroup.RESILIENCY])

        with mock.patch("redis.observability.metrics.OTEL_AVAILABLE", True):
            collector = RedisMetricsCollector(mock_meter, config)

        with mock.patch.object(
            recorder, "_get_or_create_collector", return_value=collector
        ):
            pipeline = Pipeline(
                connection_pool=mock_connection_pool,
                response_callbacks={},
                transaction=False,
                shard_hint=None,
            )

            # Make pipeline always fail
            pipeline._execute_pipeline = mock.MagicMock(
                side_effect=redis.ConnectionError("Connection failed")
            )
            pipeline.command_stack = [
                (("SET", "key1", "value1"), {}),
            ]

            # Execute pipeline - should fail after all retries
            with pytest.raises(redis.ConnectionError):
                pipeline.execute()

            # Verify histogram.record() was called 2 times (for retry attempts only)
            # Final error does NOT call record_operation_duration
            assert self.operation_duration.record.call_count == 2

            calls = self.operation_duration.record.call_args_list

            # Both retry calls should have error.type
            for call in calls:
                assert "error.type" in call[1]["attributes"]
                assert call[1]["attributes"]["db.operation.name"] == "PIPELINE"

            # Verify record_error_count was called once for the final error
            self.client_errors.add.assert_called_once()
            error_call_args = self.client_errors.add.call_args
            error_attrs = error_call_args[1]["attributes"]
            assert "error.type" in error_attrs

        recorder.reset_collector()


class _FakeRetry:
    """Models a zero-retry Retry: run the callable once, invoke the failure
    callback with failure_count=1 on a connection error, then surface it."""

    def get_retries(self):
        return 0

    def call_with_retry(
        self, do, fail, is_retryable=None, with_failure_count=False
    ):
        try:
            return do()
        except (redis.ConnectionError, redis.TimeoutError, OSError) as error:
            if with_failure_count:
                fail(error, 1)
            else:
                fail(error)
            raise


class FakePolicyConnection:
    """In-memory connection: records immediate sends and packed batches and
    replays scripted responses without touching a socket."""

    def __init__(self, responses=(), fail_command_names=(), fail_packed=False):
        self.host = "localhost"
        self.port = 6379
        self.db = 0
        self.retry = _FakeRetry()
        self._responses = list(responses)
        self._fail_command_names = set(fail_command_names)
        self._fail_packed = fail_packed
        self.immediate = []
        self.packs = []
        self.disconnects = 0

    def pack_commands(self, args):
        batch = list(args)
        self.packs.append(batch)
        return batch

    def send_packed_command(self, packed):
        if self._fail_packed:
            raise redis.ConnectionError("boom on packed write")

    def send_command(self, *args, **kwargs):
        self.immediate.append(args)
        if args[0] in self._fail_command_names:
            raise redis.ConnectionError(f"boom on {args[0]}")

    def read_response(self, *args, **kwargs):
        return self._responses.pop(0)

    def disconnect(self, *args, **kwargs):
        self.disconnects += 1


class FakePolicyPool:
    def __init__(self, connection):
        self.connection = connection
        self.released = []

    def get_connection(self, *args, **kwargs):
        return self.connection

    def release(self, connection):
        self.released.append(connection)

    def get_encoder(self):
        return Encoder(encoding="utf-8", encoding_errors="strict", decode_responses=False)


def make_policy_pipeline(transaction=True, conn=None):
    conn = conn or FakePolicyConnection()
    pool = FakePolicyPool(conn)
    return Pipeline(pool, {}, transaction, None), conn, pool


def packed_names(conn):
    return [args[0] for args in conn.packs[-1]]


def immediate_names(conn):
    return [args[0] for args in conn.immediate]


class TestPipelinePolicyStack:
    """Serverless tests for the per-Pipeline transaction/watch policy stack."""

    def test_empty_stack_matches_pin(self):
        pipe, conn, pool = make_policy_pipeline(transaction=True)
        pin = EffectivePipelinePolicy.merge([])
        assert pipe.pipeline_policy_profile == pin
        profile = pipe.pipeline_policy_profile
        assert profile.route_multi_exec(True, False) is True
        assert profile.route_multi_exec(False, False) is False
        assert profile.dispatch_immediately(True, "GET", False) is True
        assert profile.dispatch_immediately(False, "GET", False) is False
        assert profile.dispatch_immediately(False, "WATCH", False) is True
        assert profile.dispatch_immediately(False, "WATCH", True) is False
        assert profile.preload_scripts is True
        assert profile.script_buffering_forbidden is False
        assert profile.raise_watch_error_on_break is True
        assert profile.reraise_connection_errors is False
        assert profile.release_connection_on_empty is False
        assert profile.execute_forbidden is False
        assert profile.watch_forbidden is False

    def test_multi_exec_frame_packs_multi_exec(self):
        # Pipeline created without transactions; the frame forces MULTI/EXEC.
        conn = FakePolicyConnection([b"OK", b"QUEUED", b"QUEUED", [True, b"v"]])
        pipe, conn, pool = make_policy_pipeline(transaction=False, conn=conn)
        with pipe.push_pipeline_policy(exec_mode=ExecMode.MULTI_EXEC):
            pipe.set("a", 1).get("a")
            assert pipe.execute() == [True, b"v"]
        assert packed_names(conn) == ["MULTI", "SET", "GET", "EXEC"]

    def test_bare_pipeline_frame_omits_multi_exec(self):
        # Pipeline created with transactions; the frame forces a bare pack.
        conn = FakePolicyConnection([True, b"v"])
        pipe, conn, pool = make_policy_pipeline(transaction=True, conn=conn)
        with pipe.push_pipeline_policy(exec_mode=ExecMode.BARE_PIPELINE):
            pipe.set("a", 1).get("a")
            assert pipe.execute() == [True, b"v"]
        assert packed_names(conn) == ["SET", "GET"]

    def test_forbid_execute_frame_blocks_execute(self):
        conn = FakePolicyConnection()
        pipe, conn, pool = make_policy_pipeline(conn=conn)
        with pipe.push_pipeline_policy(exec_mode=ExecMode.FORBID_EXECUTE):
            pipe.set("a", 1)
            with pytest.raises(redis.PipelinePolicyError):
                pipe.execute()
        assert conn.packs == []
        # nothing was borrowed, nothing to release
        assert pool.released == []

    def test_watch_is_immediate_under_immediate_pre_multi(self):
        conn = FakePolicyConnection([b"OK", b"v", b"OK", b"QUEUED", [True]])
        pipe, conn, pool = make_policy_pipeline(conn=conn)
        with pipe.push_pipeline_policy(
            watch_dispatch=WatchDispatch.IMMEDIATE_PRE_MULTI
        ):
            pipe.watch("x")
            assert immediate_names(conn) == ["WATCH"]
            assert pipe.command_stack == []
            assert pipe.watching is True
            # reading while watching stays immediate and unbuffered
            assert pipe.get("a") == b"v"
            assert immediate_names(conn) == ["WATCH", "GET"]
            pipe.multi()
            pipe.set("a", 2)
            assert pipe.execute() == [True]
        assert packed_names(conn) == ["MULTI", "SET", "EXEC"]

    def test_watch_is_buffered_under_stack_after_multi(self):
        # one OK for the queued WATCH reply, one for reset()'s UNWATCH
        conn = FakePolicyConnection([b"OK", b"OK"])
        pipe, conn, pool = make_policy_pipeline(transaction=False, conn=conn)
        with pipe.push_pipeline_policy(
            watch_dispatch=WatchDispatch.STACK_AFTER_MULTI
        ):
            pipe.watch("x")
            assert immediate_names(conn) == []
            assert pipe.command_stack[0][0] == ("WATCH", "x")
            assert pipe.watching is False
            assert pipe.execute() == [b"OK"]
        assert packed_names(conn) == ["WATCH"]

    def test_watching_read_buffers_under_stack_after_multi(self):
        conn = FakePolicyConnection()
        pipe, conn, pool = make_policy_pipeline(transaction=False, conn=conn)
        pipe.watching = True
        with pipe.push_pipeline_policy(
            watch_dispatch=WatchDispatch.STACK_AFTER_MULTI
        ):
            pipe.get("a")
        assert immediate_names(conn) == []
        assert pipe.command_stack[0][0] == ("GET", "a")

    def test_forbid_watch_frame(self):
        conn = FakePolicyConnection()
        pipe, conn, pool = make_policy_pipeline(conn=conn)
        with pipe.push_pipeline_policy(watch_dispatch=WatchDispatch.FORBID_WATCH):
            with pytest.raises(redis.PipelinePolicyError):
                pipe.watch("x")
            with pytest.raises(redis.PipelinePolicyError):
                pipe.execute_command("WATCH", "x")
            # ordinary commands remain bufferable
            pipe.set("a", 1)
            assert pipe.command_stack[0][0] == ("SET", "a", 1)
        assert immediate_names(conn) == []

    def test_script_preload_exists_then_load_immediate_not_in_multi_stack(self):
        conn = FakePolicyConnection(
            [[0], "fake-sha", b"OK", b"QUEUED", [42]]
        )
        pipe, conn, pool = make_policy_pipeline(conn=conn)
        script = Script(pipe, "return 42")
        with pipe.push_pipeline_policy(
            script_preload=ScriptPreload.EXISTS_THEN_LOAD
        ):
            script(keys=[], args=[], client=pipe)
            assert pipe.execute() == [42]
            # frames outlive execute()/reset()
            assert len(pipe.pipeline_policy_stack) == 1
        assert immediate_names(conn) == ["SCRIPT EXISTS", "SCRIPT LOAD"]
        assert packed_names(conn) == ["MULTI", "EVALSHA", "EXEC"]

    def test_script_skip_preload_frame_skips_exists_load(self):
        conn = FakePolicyConnection([b"OK", b"QUEUED", [42]])
        pipe, conn, pool = make_policy_pipeline(conn=conn)
        script = Script(pipe, "return 42")
        with pipe.push_pipeline_policy(
            script_preload=ScriptPreload.SKIP_PRELOAD
        ):
            script(keys=[], args=[], client=pipe)
            assert pipe.execute() == [42]
        assert immediate_names(conn) == []
        assert packed_names(conn) == ["MULTI", "EVALSHA", "EXEC"]

    def test_forbid_buffer_script_blocks_buffered_script_commands(self):
        conn = FakePolicyConnection()
        pipe, conn, pool = make_policy_pipeline(conn=conn)
        with pipe.push_pipeline_policy(
            script_preload=ScriptPreload.FORBID_BUFFER_SCRIPT
        ):
            with pytest.raises(redis.PipelinePolicyError):
                pipe.pipeline_execute_command("SCRIPT LOAD", "return 1")
            with pytest.raises(redis.PipelinePolicyError):
                pipe.execute_command("SCRIPT EXISTS", "sha")
            assert pipe.command_stack == []
        # without a frame, buffering script commands stays possible
        pipe.pipeline_execute_command("SCRIPT LOAD", "return 1")
        assert pipe.command_stack[0][0] == ("SCRIPT LOAD", "return 1")

    def test_forbid_buffer_script_still_preloads_immediately(self):
        conn = FakePolicyConnection(
            [[0], "fake-sha", b"OK", b"QUEUED", [42]]
        )
        pipe, conn, pool = make_policy_pipeline(conn=conn)
        script = Script(pipe, "return 42")
        with pipe.push_pipeline_policy(
            script_preload=ScriptPreload.FORBID_BUFFER_SCRIPT
        ):
            script(keys=[], args=[], client=pipe)
            assert pipe.execute() == [42]
        assert immediate_names(conn) == ["SCRIPT EXISTS", "SCRIPT LOAD"]
        assert packed_names(conn) == ["MULTI", "EVALSHA", "EXEC"]

    def test_exec_none_raises_watch_error(self):
        conn = FakePolicyConnection([b"OK", b"QUEUED", None])
        pipe, conn, pool = make_policy_pipeline(conn=conn)
        pipe.set("a", 1)
        with pytest.raises(redis.WatchError):
            pipe.execute()
        assert pool.released == [conn]

    def test_exec_none_raises_watch_error_under_watch_error_frame(self):
        conn = FakePolicyConnection([b"OK", b"QUEUED", None])
        pipe, conn, pool = make_policy_pipeline(conn=conn)
        with pipe.push_pipeline_policy(
            watch_fault=WatchFault.WATCH_ERROR_ON_BREAK
        ):
            pipe.set("a", 1)
            with pytest.raises(redis.WatchError):
                pipe.execute()

    def test_immediate_break_while_watching_raises_watch_error(self):
        conn = FakePolicyConnection(
            [b"OK", b"OK"], fail_command_names=("GET",)
        )
        pipe, conn, pool = make_policy_pipeline(conn=conn)
        pipe.watch("x")
        with pytest.raises(redis.WatchError):
            pipe.get("a")
        assert conn.disconnects == 1
        # the failed GET is recorded before the socket error; reset() adds UNWATCH
        assert immediate_names(conn) == ["WATCH", "GET", "UNWATCH"]
        assert pool.released == [conn]

    def test_immediate_break_reraises_raw_connection_error(self):
        conn = FakePolicyConnection(
            [b"OK", b"OK"], fail_command_names=("GET",)
        )
        pipe, conn, pool = make_policy_pipeline(conn=conn)
        pipe.watch("x")
        with pipe.push_pipeline_policy(
            watch_fault=WatchFault.RERAISE_CONNECTION
        ):
            with pytest.raises(redis.ConnectionError):
                pipe.get("a")
        assert conn.disconnects == 1
        # no reset/unwatch rewriting the failure path
        assert immediate_names(conn) == ["WATCH", "GET"]
        assert pool.released == []

    def test_execute_break_while_watching_raises_watch_error(self):
        conn = FakePolicyConnection([b"OK", b"OK"], fail_packed=True)
        pipe, conn, pool = make_policy_pipeline(conn=conn)
        pipe.watch("x")
        pipe.multi()
        pipe.set("a", 1)
        with pytest.raises(redis.WatchError):
            pipe.execute()
        assert conn.disconnects == 1
        assert immediate_names(conn) == ["WATCH", "UNWATCH"]
        assert pool.released == [conn]

    def test_execute_break_reraises_raw_connection_error(self):
        conn = FakePolicyConnection([b"OK", b"OK"], fail_packed=True)
        pipe, conn, pool = make_policy_pipeline(conn=conn)
        pipe.watch("x")
        pipe.multi()
        pipe.set("a", 1)
        with pipe.push_pipeline_policy(
            watch_fault=WatchFault.RERAISE_CONNECTION
        ):
            with pytest.raises(redis.ConnectionError):
                pipe.execute()
        assert conn.disconnects == 1
        assert pool.released == [conn]

    def test_reset_unwatch_error_reraised_under_reraise_frame(self):
        conn = FakePolicyConnection(fail_command_names=("UNWATCH",))
        pipe, conn, pool = make_policy_pipeline(conn=conn)
        pipe.watching = True
        pipe.connection = conn
        # historical behavior: reset() swallows the connection error
        pipe.reset()
        assert conn.disconnects == 1
        assert pool.released == [conn]

        conn2 = FakePolicyConnection(fail_command_names=("UNWATCH",))
        pipe2, conn2, pool2 = make_policy_pipeline(conn=conn2)
        pipe2.watching = True
        pipe2.connection = conn2
        with pipe2.push_pipeline_policy(
            watch_fault=WatchFault.RERAISE_CONNECTION
        ):
            with pytest.raises(redis.ConnectionError):
                pipe2.reset()
        assert conn2.disconnects == 1

    def test_empty_stack_short_circuit_keeps_held_connection(self):
        conn = FakePolicyConnection()
        pipe, conn, pool = make_policy_pipeline(conn=conn)
        pipe.connection = conn
        assert pipe.execute() == []
        assert pool.released == []
        assert pipe.connection is conn
        with pipe.push_pipeline_policy(
            stack_empty=StackEmptyMode.SHORT_CIRCUIT
        ):
            assert pipe.execute() == []
        assert pool.released == []
        assert pipe.connection is conn

    def test_require_noop_reset_releases_held_connection(self):
        conn = FakePolicyConnection()
        pipe, conn, pool = make_policy_pipeline(conn=conn)
        pipe.connection = conn
        with pipe.push_pipeline_policy(
            stack_empty=StackEmptyMode.REQUIRE_NOOP_RESET
        ):
            assert pipe.execute() == []
        assert pool.released == [conn]
        assert pipe.connection is None

    def test_nested_frames_change_behavior_and_restore_outer(self):
        conn = FakePolicyConnection(
            [True, b"OK", b"QUEUED", b"QUEUED", [True, b"v"]]
        )
        pipe, conn, pool = make_policy_pipeline(transaction=True, conn=conn)
        outer = pipe.push_pipeline_policy(exec_mode=ExecMode.MULTI_EXEC)
        try:
            with pipe.push_pipeline_policy(exec_mode=ExecMode.BARE_PIPELINE):
                pipe.set("a", 1)
                assert pipe.execute() == [True]
                assert packed_names(conn) == ["SET"]
            # inner pop restores the outer multi_exec profile
            assert pipe.pipeline_policy_profile.exec_mode is ExecMode.MULTI_EXEC
            assert pipe.pipeline_policy_profile.route_multi_exec(False, False)
            pipe.set("a", 1).get("a")
            assert pipe.execute() == [True, b"v"]
            assert packed_names(conn) == ["MULTI", "SET", "GET", "EXEC"]
        finally:
            outer.pop()
        assert len(pipe.pipeline_policy_stack) == 0
        assert pipe.pipeline_policy_profile == EffectivePipelinePolicy.merge([])

    def test_frame_pops_cleanly_on_exception(self):
        pipe, conn, pool = make_policy_pipeline(transaction=False)
        with pytest.raises(ValueError):
            with pipe.push_pipeline_policy(exec_mode=ExecMode.BARE_PIPELINE):
                raise ValueError("middleware boom")
        assert len(pipe.pipeline_policy_stack) == 0
        assert pipe.pipeline_policy_profile == EffectivePipelinePolicy.merge([])

    def test_nested_dispatch_frame_immediate_restores_buffering(self):
        conn = FakePolicyConnection([b"OK", b"OK"])
        pipe, conn, pool = make_policy_pipeline(transaction=False, conn=conn)
        with pipe.push_pipeline_policy(
            watch_dispatch=WatchDispatch.STACK_AFTER_MULTI
        ):
            pipe.watch("x")
            assert pipe.command_stack[0][0] == ("WATCH", "x")
            pipe.reset()
            # reset() must not pop middleware frames
            assert len(pipe.pipeline_policy_stack) == 1
            with pipe.push_pipeline_policy(
                watch_dispatch=WatchDispatch.IMMEDIATE_PRE_MULTI
            ):
                pipe.watch("y")
                assert immediate_names(conn) == ["WATCH"]
            pipe.watch("z")
            assert pipe.command_stack[0][0] == ("WATCH", "z")

    def test_lifo_pop_order_enforced(self):
        pipe, conn, pool = make_policy_pipeline()
        outer = pipe.push_pipeline_policy(exec_mode=ExecMode.MULTI_EXEC)
        pipe.push_pipeline_policy(exec_mode=ExecMode.BARE_PIPELINE)
        with pytest.raises(RuntimeError):
            outer.pop()
        pipe.pipeline_policy_stack.pop()
        outer.pop()
        assert len(pipe.pipeline_policy_stack) == 0

    def test_invalid_frame_value_rejected(self):
        pipe, conn, pool = make_policy_pipeline()
        with pytest.raises(TypeError):
            pipe.push_pipeline_policy(exec_mode="multi_exec")
        assert len(pipe.pipeline_policy_stack) == 0

    def test_executor_guards_disagree_with_frame(self):
        conn = FakePolicyConnection()
        pipe, conn, pool = make_policy_pipeline(conn=conn)
        commands = [(("SET", "a", 1), {})]
        with pipe.push_pipeline_policy(exec_mode=ExecMode.BARE_PIPELINE):
            with pytest.raises(redis.PipelinePolicyError):
                pipe._execute_transaction(conn, commands, True)
        with pipe.push_pipeline_policy(exec_mode=ExecMode.MULTI_EXEC):
            with pytest.raises(redis.PipelinePolicyError):
                pipe._execute_pipeline(conn, commands, True)
        # a raw MULTI buffered under bare_pipeline is rejected at execution
        with pipe.push_pipeline_policy(exec_mode=ExecMode.BARE_PIPELINE):
            pipe.execute_command("MULTI")
            with pytest.raises(redis.PipelinePolicyError):
                pipe.execute()

    def test_two_pipelines_do_not_share_stack(self):
        pipe_a, conn_a, pool_a = make_policy_pipeline()
        pipe_b, conn_b, pool_b = make_policy_pipeline()
        with pipe_a.push_pipeline_policy(exec_mode=ExecMode.FORBID_EXECUTE):
            assert pipe_b.pipeline_policy_profile == EffectivePipelinePolicy.merge(
                []
            )
            assert pipe_b.pipeline_policy_profile.execute_forbidden is False
            assert pipe_a.pipeline_policy_profile.execute_forbidden is True
        assert pipe_b.pipeline_policy_profile == EffectivePipelinePolicy.merge([])
