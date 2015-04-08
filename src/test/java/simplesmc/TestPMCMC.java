package simplesmc;

import java.util.List;
import java.util.Random;

import simplesmc.hmm.HMMProposal;
import simplesmc.hmm.HMMUtils;
import simplesmc.hmm.ToyHMMParams;
import simplesmc.pmcmc.PMCMCFactor;
import bayonet.distributions.Uniform;
import bayonet.distributions.Uniform.MinMaxParameterization;
import blang.MCMCAlgorithm;
import blang.MCMCFactory;
import blang.annotations.DefineFactor;
import blang.processing.Processor;
import blang.processing.ProcessorContext;
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
public class TestPMCMC implements Runnable, Processor
{
  @OptionSet(name = "factory")
  public final MCMCFactory factory = new MCMCFactory();
  
  @Option
  public int nStates = 2;
  
  @Option
  public double trueSelfTransitionPr = 0.8;
  
  @Option
  public int observationLength = 100;
  
  @Option
  public Random generateRandom = new Random(1);
  
  @OptionSet(name = "smc")
  public SMCOptions options = new SMCOptions();
  
  private List<Integer> observations;
  
  public class Model
  {
    public ToyHMMParams hmmParams = new ToyHMMParams(nStates);

    @DefineFactor
    public PMCMCFactor<Integer> likelihood = new PMCMCFactor<>(hmmParams, new SMCAlgorithm<>(new HMMProposal(hmmParams, observations), options));
     
    @DefineFactor
    public Uniform<MinMaxParameterization> prior = Uniform.on(hmmParams.selfTransitionProbability);
  }
  
  // Note: only instantiate this in run() to avoid problems with command line argument parsing
  public Model model;

  @Override
  public void run()
  {
    factory.addProcessor(this);
    ToyHMMParams trueParams = new ToyHMMParams(nStates);
    trueParams.selfTransitionProbability.setValue(trueSelfTransitionPr);
    observations = HMMUtils.generate(generateRandom, trueParams, observationLength).getRight();
    model = new Model();
    MCMCAlgorithm mcmc = factory.build(model, false);
    mcmc.options.CODA = true;
    System.out.println(mcmc.model);
    mcmc.run();
  }
   
  public static void main(String [] args)
  {
    Mains.instrumentedRun(args, new TestPMCMC());
  }

  @Override
  public void process(ProcessorContext context)
  {
    
  }
}
