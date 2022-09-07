use super::*;

mod data {
	use super::*;

	#[test]
	fn basic() {
		let arena = Arena::new();
		let mut graph = RenderGraph::new();

		for _ in 0..2 {
			let mut frame = graph.frame(&arena);

			struct Data<'graph>(Vec<usize, &'graph Arena>);

			let mut p1 = frame.pass("Pass 1");
			let (set, get) = p1.data_output::<Data>();
			p1.build(|mut ctx| {
				let mut v = Vec::new_in(ctx.arena());
				v.push(1);
				v.push(2);
				ctx.set_data(set, Data(v));
			});

			let mut p2 = frame.pass("Pass 2");
			p2.data_input(&get);
			p2.build(|ctx| {
				let data = ctx.get_data(get);
				assert_eq!(data.0, vec![1, 2]);
			});

			frame.run();
		}
	}

	#[test]
	#[should_panic]
	fn try_access_data_from_previous_frame() {
		let arena = Arena::new();
		let mut graph = RenderGraph::new();

		struct Data<'graph>(Vec<usize, &'graph Arena>);

		let mut id: Option<RefId<Data>> = None;
		for _ in 0..2 {
			let mut frame = graph.frame(&arena);

			let mut p1 = frame.pass("Pass 1");
			let (set, get) = p1.data_output::<Data>();
			p1.build(|mut ctx| {
				let mut v = Vec::new_in(ctx.arena());
				v.push(1);
				v.push(2);
				ctx.set_data(set, Data(v));
			});

			let ref_id = get.to_ref();

			let mut p2 = frame.pass("Pass 2");
			p2.data_input_ref(ref_id);
			p2.build(|ctx| {
				let data = ctx.get_data_ref(ref_id);
				assert_eq!(data.0, vec![1, 2]);
				if let Some(id) = id {
					let _ = ctx.get_data_ref(id);
				}
			});

			frame.run();

			id = Some(ref_id);
		}
	}
}
