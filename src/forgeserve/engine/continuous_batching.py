
from __future__ import annotations

import uuid

from forgeserve.engine.response import GenerationResponse
from forgeserve.logger import get_logger
from forgeserve.scheduler.continuous_batching import ContinuousBatchScheduler
from forgeserve.scheduler.request import RequestState


logger = get_logger(__name__)


class ContinuousBatchingEngine:
    """
    Public orchestration layer for continuous batching.

    Engine: request submission, execution loop, results.
    Scheduler: admission, scheduling, request lifecycle.
    Executor: prefill, sampling, model execution.
    Runtime: model and KV-cache operations.
    """

    def __init__(
        self,
        scheduler: ContinuousBatchScheduler,
    ) -> None:
        self.scheduler = scheduler
        self.runtime = scheduler.runtime

        # Keep references to requests, including completed ones.
        self._requests: dict[str, RequestState] = {}

        # Generated output token IDs for each request.
        self._generated_ids: dict[str, list[int]] = {}

        logger.info(
            "ContinuousBatchingEngine initialized: "
            "max_batch_size=%s total_blocks=%d",
            scheduler.max_batch_size,
            scheduler.block_manager.num_blocks,
        )

    def submit(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        request_id: str | None = None,
    ) -> RequestState:

        if request_id is None:
            request_id = uuid.uuid4().hex

        if request_id in self._requests:
            raise ValueError(
                f"Request '{request_id}' already exists."
            )

        request = self.scheduler.add_request(
            request_id=request_id,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )

        self._requests[request_id] = request
        self._generated_ids[request_id] = []

        logger.info(
            "Engine request submitted: id=%s max_tokens=%d",
            request_id,
            max_new_tokens,
        )

        logger.debug(
            "Request queued: id=%s waiting=%d running=%d",
            request_id,
            self.scheduler.num_waiting,
            self.scheduler.num_running,
        )

        return request

    def step(self) -> int:
        """
        Execute one scheduler iteration and collect new tokens.
        """

        # Capture state before execution.
        previous = {
            request_id: (
                request.generated_tokens,
                request.is_finished,
            )
            for request_id, request in self._requests.items()
        }

        logger.debug(
            "Engine step started: state=%s",
            self.scheduler.snapshot(),
        )

        try:
            processed = self.scheduler.step()

            for request_id, request in self._requests.items():

                old_count, was_finished = previous[request_id]

                new_tokens = request.generated_tokens - old_count

                if new_tokens not in (0, 1):
                    raise RuntimeError(
                        f"Unexpected token count change for "
                        f"request '{request_id}': {new_tokens}"
                    )

                if new_tokens == 1:
                    if request.last_token_id is None:
                        raise RuntimeError(
                            f"Missing token ID for request '{request_id}'"
                        )

                    self._generated_ids[request_id].append(
                        request.last_token_id
                    )

                    logger.debug(
                        "Token collected: id=%s token=%d total=%d",
                        request_id,
                        request.last_token_id,
                        request.generated_tokens,
                    )

                # Emit completion once, not on every future step.
                if request.is_finished and not was_finished:
                    logger.info(
                        "Engine request completed: "
                        "id=%s generated=%d reason=%s "
                        "ttft_ms=%s latency_ms=%s",
                        request_id,
                        request.generated_tokens,
                        request.finish_reason,
                        request.time_to_first_token_ms,
                        request.total_latency_ms,
                    )

            logger.debug(
                "Engine step completed: processed=%d state=%s",
                processed,
                self.scheduler.snapshot(),
            )

            return processed

        except Exception:
            logger.exception(
                "Engine step failed: state=%s",
                self.scheduler.snapshot(),
            )
            raise

    def get_result(
        self,
        request_id: str,
    ) -> GenerationResponse:
        """
        Convert completed request tokens into a response.
        """

        request = self._requests[request_id]

        if not request.is_finished:
            raise RuntimeError(
                f"Request '{request_id}' has not finished."
            )

        token_ids = self._generated_ids[request_id]

        text = self.runtime.decode(token_ids)

        response = GenerationResponse(
            text=text,
            generated_tokens=request.generated_tokens,
            finish_reason=request.finish_reason,
        )

        logger.debug(
            "Result created: id=%s generated=%d text_length=%d",
            request_id,
            response.generated_tokens,
            len(response.text),
        )

        return response

    def run_until_idle(
        self,
    ) -> dict[str, GenerationResponse]:
        """
        Run the scheduler until no requests remain.
        """

        logger.info(
            "Engine execution started: waiting=%d running=%d",
            self.scheduler.num_waiting,
            self.scheduler.num_running,
        )

        iterations = 0

        while (
            self.scheduler.num_waiting > 0
            or self.scheduler.num_running > 0
        ):
            processed = self.step()
            iterations += 1

            if processed == 0:
                state = self.scheduler.snapshot()

                logger.error(
                    "Engine stalled: iteration=%d state=%s",
                    iterations,
                    state,
                )

                raise RuntimeError(
                    "Scheduler made no progress. "
                    "Check KV-cache capacity and prompt sizes."
                )

        results = {
            request_id: self.get_result(request_id)
            for request_id in self._requests
        }

        logger.info(
            "Engine execution completed: "
            "iterations=%d total_requests=%d state=%s",
            iterations,
            len(results),
            self.scheduler.snapshot(),
        )

        return results

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
    ) -> GenerationResponse:
        """
        Submit and execute a single request synchronously.
        """

        request = self.submit(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )

        results = self.run_until_idle()

        return results[request.request_id]
