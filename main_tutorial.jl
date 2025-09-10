using Agents
using Random

# Step 1: Creating the space

# A non-periodic 2D grid, where only one agent per position is allowed
# is the appropriate space for Schelling's Model
# Chebyshev metric considers 8 neighbors around a grid position
size = (12, 12)
space = GridSpaceSingle(size, periodic = false, metric = :chebyshev)

# Step 2: Creating the agent type

@agent struct SchellingAgent(GridAgent{2})
    mood::Bool      # Happy (true) or unhappy (false) in its position
    group::Symbol   # :red or :green, determines mood based on its neighbors
end

# Explicitly listing all SchellingAgent data structure fields (names and types)
for (name, type) in zip(fieldnames(SchellingAgent), fieldtypes(SchellingAgent))
    println(name, "::", type)
end

# Step 3: Defining the evolution rules

function schelling_step!(agent, model)
    min_happy = model.min_to_be_happy
    count_neighbor_same_group = 0

    for neighbor in nearby_agents(agent, model)
        if agent.group == neighbor.group
            count_neighbor_same_group += 1
        end
    end

    if count_neighbor_same_group < min_happy
        agent.mood = false
        move_agent_single!(agent, model)
    else
        agent.mood = true
    end
    return
end

# Step 4: Initializing the model using the AgentBasedModel structure

schelling_properties = Dict(:min_to_be_happy => 3)
# properties = Dict(:min_to_be_happy => 3) # there would be no need to repeat the keyword if using the same name

#(Optional) scheduler
# if different from the standard 'fastest' or if it matters due to using an agent_step! function
schelling_scheduler = Schedulers.ByProperty(:group)

schelling = StandardABM(
    # Input arguments
    SchellingAgent, space;
    # Keyword arguments
    properties = schelling_properties,
    agent_step! = schelling_step!,
    scheduler = schelling_scheduler, # (optional)
)

# Populating the model with agents
nagents(schelling) # checking on current number of agents

# Method 1: adding agent to specified position using (implicit) args
added_agent_1 = add_agent!((1, 1), schelling, false, :red)

# Method 2: adding agent to a randomly picked position using (implicit) args
added_agent_2 = add_agent!(schelling, false, :red)

# Method 3: adding agent to random position using (explicit) keyword args (kwargs)
added_agent_3 = add_agent!(schelling; mood = true, group = :green)

# Method 4: adding agent with function that autamically respects one agent per position restriction
added_agent_4 = add_agent_single!(schelling; mood = false, group = :red)

nagents(schelling) # checking new number of agents

# RECOMMENDED: Create a function to initialize and populate the model
function initialize(; num_agents = 320, gridsize = (20, 20), min_to_be_happy = 3, seed = 42)
    space = GridSpaceSingle(gridsize; periodic = false, metric = :chebyshev)
    schelling_properties = Dict(:min_to_be_happy => min_to_be_happy)
    rng = Xoshiro(seed)

    model = StandardABM(
        SchellingAgent, space;
        agent_step! = schelling_step!,
        properties = schelling_properties,
        rng,
        container = Vector, # used for performance, appropriate for Schelling because population is constant
        scheduler = Schedulers.Randomly() # all agents are activated once at a random order
    )

    # Populate the model
    for i in 1:num_agents
        # The population is divided in half between the two group types
        if i < num_agents/2
            group = :red
        else
            group = :green
        end
        # All agents are unhappy at the start
        add_agent_single!(model; mood = false, group = group)
        
        # Alternative using Julia's ternary operator. Concise but less readable
        # add_agent_single!(model; mood = false, group = i < num_agents/2 ? :red : :green)
    end
    return model
end

# Function version
schelling_vf = initialize()

# Step 5: Evolve the model

# Progress the simulation for 1 step
step!(schelling_vf)

# Progress the simulation for 3 steps
step!(schelling_vf, 3)

# Progress the simulation until a given function which takes the model as input and current model time
# evaluates to true. This should be used to check for when an ending condition is metric

# Concise function definition
happy_90(model, time) = count(a -> a.mood == true, allagents(model))/nagents(model) ≥ 0.9

step!(schelling_vf, happy_90)

# Checking how many steps the agent has taken so far
abmtime(schelling_vf)

# Longer but more readable version
function happy_90(model, time)
    # count() function syntaxis: count(function, collection) <- counts how many items satisfy function condition 
    num_happy_agents = count(agent -> agent.mood == true, allagents(model))
    num_total_agents = nagents(model)
    fraction_happy = num_happy_agents/num_total_agents
    
    return fraction_happy ≥ 0.9 || time ≥ 1000 # Included failsafe in form of number of time steps boundary
end

# Step 6: Visualizations

using CairoMakie # Mostly for regular 2D plots

groupcolor(a) = a.group == :red ? :red : :green
groupmarker(a) = a.group == :red ? :circle : :rect

# Generating figure with final model state
figure, axis = abmplot(schelling_vf; agent_color = groupcolor, agent_marker = groupmarker, agent_size = 10)
figure
# display(figure)

# Move to directory where the current .jl file lives
current_dir = @__DIR__
cd(current_dir)
pwd()

# Create "plots" subfolder, figure filename and path
mkpath("plots") 
figure_filename = "final_schelling_state.png"
figure_path = joinpath(current_dir, "plots", figure_filename)

# Saving figure with final model state
save(figure_path, figure)

# Initializing new model and saving evolution as video
schelling_vid_v1 = initialize(
    ; num_agents = 2300, gridsize = (50, 50), min_to_be_happy = 4, seed = 42
    )

plots_dir = joinpath(current_dir, "plots")
video_filename = "schelling_vid_v1.mp4"
video_path = joinpath(plots_dir, video_filename)

abmvideo(
    video_path, schelling_vid_v1;
    agent_color = groupcolor, agent_marker = groupmarker, 
    agent_size = 10,
    framerate = 10, frames = 100,
    title = "Schelling's segregation model, 320 agents, 20x20 grid"
)

# Checking for final number of iterations
abmtime(schelling_vid_v1)

# Simple interactive GUI
using GLMakie # GLMakie should be used instead of CairoMakie to use the interactive aspects of the plots

schelling_vint = initialize()

fig, ax, abmobs = abmplot(
    schelling_vint; add_controls = true,
    agent_color = groupcolor, agent_marker = groupmarker, agent_size = 10
)

display(fig)

# Step 7: Data collection
# run! -> running the model and collecting data while it runs
# paramscan -> data collection while scanning ranges of the parameters of the model
# ensemblerun! -> ensemble simulations and data collection

# Properties to be collected as data directly using Symbols
agent_data_properties = [:pos, :mood, :group]

schelling_vdata = initialize() # Initializing new model
agent_df, model_df = run!(schelling_vdata, 5; adata = agent_data_properties) # Run and collecta data for 5 steps
agent_df[end-10:end, :] # Display only the last few rows

# Properties to be collected as functions applied to agent Data
x(agent) = agent.pos[1] # Function to get the agent's x-coordinate
y(agent) = agent.pos[2] # Function to get the agent's y-coordinate

schelling_vfdata = initialize()
agent_data = [x, y, :mood, :group]

agent_df, model_df = run!(schelling_vfdata, 5; adata=agent_data)
agent_df[end-10:end, :]

# Collecting aggregated data for the agents
using Statistics: mean  # Must be called into scope, not originally available
using LinearAlgebra # For norm

schelling_vagg = initialize()
# Data to collect -> total number of happy agents (:mood=true), mean distance to center
r(agent) = norm(collect(agent.pos))
agent_data = [(:mood, sum), (r, mean)]
agent_df, model_df = run!(schelling_vagg, 5; adata=agent_data)
agent_df