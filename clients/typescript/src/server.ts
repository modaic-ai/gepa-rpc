import Fastify from "fastify";
import type { GEPA, GEPANode } from "./gepa";
import type {
  Example,
  EvaluateRequest,
  MakeReflectiveDatasetRequest,
  FinalizeRequest,
} from "./models";

export function createServer(gepa: GEPA, node: GEPANode) {
  const fastify = Fastify({
    connectionTimeout: 0, // time to receive full request
    keepAliveTimeout: 0, // keep-alive between requests
  });

  fastify.post("/evaluate", async (request, reply) => {
    request.raw.setTimeout(0);
    reply.raw.setTimeout(0);

    const { batch, candidate, capture_traces } =
      request.body as EvaluateRequest;

    if (!gepa.metric) {
      reply
        .status(500)
        .send({ error: "Metric not configured on GEPA instance" });
      return;
    }

    const response = await node.evaluate(
      batch,
      candidate,
      capture_traces,
      gepa.options.numThreads || 4,
      gepa.metric,
      gepa.dataset ? gepa.dataset.getInputs.bind(gepa.dataset) : undefined
    );

    return response;
  });

  fastify.post("/make_reflective_dataset", async (request, reply) => {
    request.raw.setTimeout(0);
    reply.raw.setTimeout(0);

    const { candidate, eval_batch, components_to_update } =
      request.body as MakeReflectiveDatasetRequest;

    const reflective_dataset = await gepa.make_reflective_dataset(
      candidate,
      eval_batch,
      components_to_update
    );

    return reflective_dataset;
  });

  fastify.post("/finalize", async (request, reply) => {
    const { best_candidate, results } = request.body as FinalizeRequest;
    await gepa.finalize(best_candidate, results);
    return { status: "ok" };
  });

  return fastify;
}
