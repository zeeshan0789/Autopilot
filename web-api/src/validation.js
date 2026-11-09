import { z } from "zod";

export const loginSchema = z.object({
  email: z.string().trim().toLowerCase().email().max(255),
  password: z.string().min(1).max(200),
});

export const createKeySchema = z.object({
  name: z.string().trim().min(1).max(100),
  rate_limit_per_min: z.number().int().min(1).max(100000).default(60),
});

export const updateKeySchema = z.object({
  rate_limit_per_min: z.number().int().min(1).max(100000),
});

export const createPromptSchema = z.object({
  name: z.string().regex(/^[a-z0-9][a-z0-9_]{0,63}$/, "use lowercase letters, digits and underscores"),
  description: z.string().max(500).optional(),
});

export const createVersionSchema = z.object({
  system_prompt: z.string().min(1).max(20000),
  user_template: z.string().max(20000).optional(),
  weight: z.number().int().min(0).max(1000).default(0),
  notes: z.string().max(1000).optional(),
});

export const weightsSchema = z.object({
  weights: z.record(z.string().regex(/^\d+$/, "keys are version numbers"), z.number().int().min(0).max(1000)),
});

export const hoursSchema = z.coerce.number().int().min(1).max(24 * 90).default(24);
export const bucketSchema = z.enum(["hour", "day"]).default("hour");
export const dimensionSchema = z.enum(["provider", "model", "cache", "route", "status"]);

export const tracesQuerySchema = z.object({
  limit: z.coerce.number().int().min(1).max(200).default(50),
  offset: z.coerce.number().int().min(0).default(0),
  status: z.enum(["success", "error"]).optional(),
});