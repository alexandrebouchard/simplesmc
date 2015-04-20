package simplesmc;

import java.util.List;
import java.util.Random;

import simplesmc.hmm.HMMProblemSpecification;
import simplesmc.hmm.HMMUtils;
import simplesmc.hmm.ToyHMMParams;
import simplesmc.pmcmc.PMCMCFactor;
import tutorialj.Tutorial;
import bayonet.distributions.Uniform;
import bayonet.distributions.Uniform.MinMaxParameterization;
import blang.MCMCAlgorithm;
import blang.MCMCFactory;
import blang.annotations.DefineFactor;
import briefj.opt.Option;
import briefj.opt.OptionSet;
import briefj.run.Mains;


/**
 * An example that shows an instrumented example of blang,
 * i.e. using command line options and nicely organized,
 * reproducible output.
 * 
 * @author Alexandre Bouchard (alexandre.bouchard@gmail.com)
 *
 */
public class TestPMCMC implements Runnable
{
  @OptionSet(name = "factory")
  public final MCMCFactory factory = new MCMCFactory();
  
  @Option(gloss = "Number of latent states")
  public int nStates = 2;
  
  @Option(gloss = "Self-transition probability used to generated the data")
  public double trueSelfTransitionPr = 0.8;
  
  @Option(gloss = "Lengths of the generated observation (i.e. number of time steps in the HMM)")
  public int observationLength = 100;
  
  @Option(gloss = "Random seed for generating the data")
  public Random generateRandom = new Random(1);
  
  @OptionSet(name = "smc")
  public SMCOptions options = new SMCOptions();
  
  private List<Integer> observations;
  
  /**
   * This class describes a model which will be sampled automatically, similarly 
   * to JAGS, except that here we define some of our own distributions and samplers.
   * 
   * More precisely, Model specifies a joint distribution via a factor graph. The 
   * fields annotated by DefineFactor below are used as the factors in the factor
   * graph. Within these factors, the fields annotated with FactorComponent and
   * FactorArgument are used as the variables (i.e. blang looks recursively in all 
   * FactorComponent fields to find the FactorArguments).
   * 
   * @author Alexandre Bouchard (alexandre.bouchard@gmail.com)
   *
   */
  public class Model
  {
    public ToyHMMParams hmmParams = new ToyHMMParams(nStates);

    /**
     * The likelihood is approximated via PMCMC ran on the HMM
     */
    @DefineFactor
    public PMCMCFactor<Integer> likelihood = 
      new PMCMCFactor<>(
        hmmParams, 
        new SMCAlgorithm<>(new HMMProblemSpecification(hmmParams, observations), options));
     
    /**
     * We put a uniform prior on the selfTransitionProbability parameter
     */
    @DefineFactor
    public Uniform<MinMaxParameterization> prior = Uniform.on(hmmParams.selfTransitionProbability);
  }
  
  // Note: only instantiate this in run() to avoid problems with command line argument parsing
  public Model model;

  /**
   * See this link for an example of a main class for a probabilistic program that samples from the posterior of a static
   * parameter using PMCMC (more precisely, only PMMH is supported at the moment). The main customization is
   * the Model class, which declaratively specifies the priors on the static parameters.
   */
  @Tutorial(showLink = true, linkPrefix = "src/test/java/", showSource = false)
  @Override
  public void run()
  {
    // generate some data
    ToyHMMParams trueParams = new ToyHMMParams(nStates);
    trueParams.selfTransitionProbability.setValue(trueSelfTransitionPr);
    observations = HMMUtils.generate(generateRandom, trueParams, observationLength).getRight();
    
    /*
     * perform posterior inference
     * this inspects all the variables defined in the model,
     * (here there is only one, selfTransitionProbability, in 
     * ToyHMMParams), looks at the declaration of their type
     * (here, RealVariable), find the types of samplers that 
     * apply to that datatype (here, those are provided for you
     * by an external library). The samplers interact with the 
     * model by calling the factors' logDensity() methods.
     */
    model = new Model();
    MCMCAlgorithm mcmc = factory.build(model, false);
    System.out.println(mcmc.model);
    mcmc.run();
  }
   
  public static void main(String [] args)
  {
    Mains.instrumentedRun(args, new TestPMCMC());
  }
}
