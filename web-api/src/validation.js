import { z } from "zod";

// Bodies are strict: a field that is not listed (a role, an owner, a status) is an error, not ignored.

export const loginSchema = z.object({
  email: z.string().trim().toLowerCase().email().max(255),
  password: z.string().min(1).max(200),
});

export const signupSchema = z
  .object({
    email: z.string().trim().toLowerCase().email().max(255),
    password: z.string().min(10, "use at least 10 characters").max(200),
    invite_code: z.string().trim().min(1).max(100).optional(),
  })
  .strict();

export const createKeySchema = z
  .object({
    name: z.string().trim().min(1).max(100),
    rate_limit_per_min: z.number().int().min(1).max(100000).default(60),
  })
  .strict();

export const updateKeySchema = z
  .object({ rate_limit_per_min: z.number().int().min(1).max(100000) })
  .strict();

export const createPromptSchema = z
  .object({
    name: z.string().regex(/^[a-z0-9][a-z0-9_]{0,63}$/, "use lowercase letters, digits and underscores"),
    description: z.string().max(500).optional(),
  })
  .strict();

export const createVersionSchema = z
  .object({
    system_prompt: z.string().min(1).max(20000),
    user_template: z.string().max(20000).optional(),
    weight: z.number().int().min(0).max(1000).default(0),
    notes: z.string().max(1000).optional(),
  })
  .strict();

export const weightsSchema = z
  .object({
    weights: z.record(z.string().regex(/^\d+$/, "keys are version numbers"), z.number().int().min(0).max(1000)),
  })
  .strict();

// What an admin may change about a member: never the role
export const updateUserSchema = z
  .object({
    status: z.enum(["active", "suspended"]).optional(),
    daily_request_limit: z.number().int().min(0).max(1_000_000).optional(),
    max_tokens_cap: z.number().int().min(1).max(100_000).optional(),
  })
  .strict()
  .refine((value) => Object.keys(value).length > 0, "nothing to change");

export const createInviteSchema = z
  .object({ note: z.string().trim().max(200).optional() })
  .strict();

export const hoursSchema = z.coerce.number().int().min(1).max(24 * 90).default(24);
export const bucketSchema = z.enum(["hour", "day"]).default("hour");
export const dimensionSchema = z.enum(["provider", "model", "cache", "route", "status"]);

export const tracesQuerySchema = z.object({
  limit: z.coerce.number().int().min(1).max(200).default(50),
  offset: z.coerce.number().int().min(0).default(0),
  status: z.enum(["success", "error"]).optional(),
});